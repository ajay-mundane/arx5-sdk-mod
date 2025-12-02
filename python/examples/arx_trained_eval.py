import math
from matplotlib import pyplot as plt
from pynput.keyboard import Key, KeyCode, Listener
from collections import defaultdict
from threading import Lock
import multiprocessing as mp
from multiprocessing.managers import SharedMemoryManager
from robotcontrol_eval import RobotControlProcess
from cameracapture_eval import CameraCaptureProcess
import pyrealsense2 as rs
from helper import ReplayBuffer, match_observations_to_actions, align_camera_timestamps, preprocess_robot_actions
import numpy as np
import pathlib
from omegaconf import OmegaConf
from tqdm import tqdm
from shared_memory.shared_memory_ring_buffer import SharedMemoryRingBuffer
from shared_memory.shared_memory_queue import SharedMemoryQueue
import time
import scipy.interpolate as si
import zmq
import cv2


class KeystrokeCounter(Listener):
    def __init__(self):
        self.key_count_map = defaultdict(lambda:0)
        self.key_press_list = list()
        self.lock = Lock()
        super().__init__(on_press=self.on_press, on_release=self.on_release)
    
    def on_press(self, key):
        with self.lock:
            self.key_count_map[key] += 1
            self.key_press_list.append(key)
    
    def on_release(self, key):
        pass
    
    def clear(self):
        with self.lock:
            self.key_count_map = defaultdict(lambda:0)
            self.key_press_list = list()
    
    def __getitem__(self, key):
        with self.lock:
            return self.key_count_map[key]
    
    def get_press_events(self):
        with self.lock:
            events = list(self.key_press_list)
            self.key_press_list = list()
            return events


class Conductor:

    def __init__(self):

        self.camera_obs_horizon = 2
        self.frequency = 10.0  # Hz
        self.camera_down_sample_steps = 1  # downsample factor for camera observations
        self.robot_obs_horizon = 2
        self.robot_down_sample_steps = 1  # downsample factor for robot observations
        self.pred_action_horizon = 10 # how many steps from prediction robot should execute before resending
        
        # Store predicted actions
        self.predicted_actions = None
        self.action_index = 1
        self.sent_action_index = 1
        self.waiting_for_reply = False
        
        # Create communication channels
        self.cmd_q_robot = mp.Queue()
        self.cmd_q_cameras = {}
        
        # Create shared memory manager for ring buffers
        self.shm_manager = SharedMemoryManager()
        self.shm_manager.start()

        # Hardware Config - load from YAML
        config_path = pathlib.Path(__file__).resolve().parent.joinpath('robotcontrol.yaml')
        rcfg = OmegaConf.load(str(config_path))
        follower_cfg = ('X5_cam', rcfg.follower.left, rcfg.follower.right)
        dof = int(rcfg.dof)
        friction_cfg = OmegaConf.to_container(rcfg.friction, resolve=True)


        # Create robot observations ring buffer
        robot_obs_examples = {
            "joint_pos_L": np.zeros(dof, dtype=np.float32),
            "gripper_pos_L": np.float32(0.0),
            "joint_pos_R": np.zeros(dof, dtype=np.float32),
            "gripper_pos_R": np.float32(0.0),
            "timestamp": np.float64(0.0)
        }
        self.robot_obs_buffer = SharedMemoryRingBuffer.create_from_examples(
            shm_manager=self.shm_manager,
            examples=robot_obs_examples,
            get_max_k=100*5,
            get_time_budget=0.01,
            put_desired_frequency=100  # Robot runs at 100Hz
        )

        # Create shared memory queue for sending over robot commands post inference
        command_examples = {
            "joint_pos_L": np.zeros(dof, dtype=np.float32),
            "gripper_pos_L": np.float32(0.0),
            "joint_pos_R": np.zeros(dof, dtype=np.float32),
            "gripper_pos_R": np.float32(0.0),
        }
        self.robot_joint_queue = SharedMemoryQueue.create_from_examples(
            shm_manager=self.shm_manager,
            examples=command_examples,
            buffer_size=256
        )

        # Start Robot Process
        robot_cfg = {"follower": follower_cfg, "dof": dof, "friction": friction_cfg}
        # NOTE: Let's also configure frequency/fps here later
        self.robot_proc = RobotControlProcess(self.cmd_q_robot, robot_cfg, self.robot_obs_buffer, self.robot_joint_queue) 
        self.robot_proc.start()

        # Start Cameras (as discussed before)
        self.camera_procs = {} # Sending over basic commands
        self.camera_obs_buffers = {} # Ring buffers for each camera's observations
        serials = self.get_connected_devices_serial()
        print(serials)
        for serial in serials:
            cmd_q_camera = mp.Queue()
            self.cmd_q_cameras[serial] = cmd_q_camera

            
            # Create camera observations ring buffer for this camera
            camera_obs_examples = {
                "color": np.zeros((480, 640, 3), dtype=np.uint8),
                "timestamp": np.float64(0.0)
            }
            camera_obs_buffer = SharedMemoryRingBuffer.create_from_examples(
                shm_manager=self.shm_manager,
                examples=camera_obs_examples,
                get_max_k=50,
                get_time_budget=0.02,
                put_desired_frequency=30  # Camera runs at 30Hz
            )
            self.camera_obs_buffers[serial] = camera_obs_buffer
            
            camera_config = {
                "resolution": (640, 480),
                "capture_fps": 30,
                "enable_color": True,
                "enable_depth": False,
                "alignment": True
                # "enable_infrared": False
            }
            # if serial in ['419522072281']:#,'317622071752']:
                # camera_config["alignment"] = False
                # Reduce bandwidth for problematic camera
                # camera_config["enable_depth"] = False
            new_camera_proc = CameraCaptureProcess(cmd_q_camera, camera_config, serial, camera_obs_buffer)
            self.camera_procs[serial] = new_camera_proc
            new_camera_proc.start()

        self.ep_idx = 0

        context = zmq.Context()
        self.socket = context.socket(zmq.REQ)
        self.socket.connect(f"tcp://localhost:8766")
        
        # # Wait for socket to connect successfully
        # connected = False
        # retry_count = 0
        # while not connected:
        #     try:
        #         self.socket.connect(f"tcp://localhost:8766")
        #         # Test connection by sending a ping
        #         self.socket.setsockopt(zmq.RCVTIMEO, 1000)  # 1 second timeout
        #         self.socket.send_string("ping")
        #         response = self.socket.recv_string()
        #         connected = True
        #         print("Successfully connected to model server!")
        #     except (zmq.error.ZMQError, zmq.Again) as e:
        #         retry_count += 1
        #         if retry_count % 5 == 0:  # Print every 5 attempts (roughly every 5 seconds)
        #             print("Waiting to connect...")
        #         time.sleep(1)
        #         # Reset socket for next attempt
        #         self.socket.close()
        #         self.socket = context.socket(zmq.REQ)
        
        # # Reset socket options after successful connection
        # self.socket.setsockopt(zmq.RCVTIMEO, -1)  # No timeout for normal operation

        self.execute = False  # Set to True to execute on robot

        # Recorder setup
        self.is_recording = False
        self.video_writer = None
        self.video_fps = 30
        self.video_path = pathlib.Path.cwd().joinpath(f"camera_stitched_{int(time.time())}.mp4")
        # Grid config: stitch all cameras horizontally or in 2xN grid depending on count
        self.stitch_rows = 1  # will adapt later based on camera count
        self.stitch_cols = max(1, len(self.camera_procs)) if hasattr(self, 'camera_procs') else 1

    def loop(self):
        with KeystrokeCounter() as key_counter:
            time.sleep(3.0)  # wait for everything to start
            command_exec = []
            try:
                running = True
                ema_alpha = 0.8
                prev_action = None
                while running:

                    #1. Get robot observations
                    #2. Get camera observations
                    #3. Align them based on timestamps
                    t = time.monotonic()
                    aligned_obs = self.get_aligned_observations()
                    # if aligned_obs:
                    #     for x,y in aligned_obs.items():
                    #         print(f"{x} got shape {y.shape}")
                    # else:
                    #     print("Failed to get aligned observations")

                    # #4. Send to model (socket based), recv actions - MAKE SURE TO WARMUP BEFOREHAND
                    if self.predicted_actions is None or self.action_index > self.pred_action_horizon:
                        inference_start = time.monotonic()
                        self.sent_action_index = self.action_index
                        # print("Request sent.")
                        self.socket.send_pyobj(aligned_obs)
                        actions = self.socket.recv_pyobj()
                        inference_time = time.monotonic() - inference_start
                        # print(f"Request received. Inference time: {inference_time:.3f}s. Approx. cycles needed={math.ceil(inference_time * self.frequency)}")
                        self.waiting_for_reply = False
                        self.action_index = 0
                    else:
                        actions = None
                    # else:
                    #     # --- STEP A: CHECK IF WE SHOULD SEND ---
                    #     trigger_send = (self.action_index - self.sent_action_index >= self.pred_action_horizon)
                        
                    #     if trigger_send and not self.waiting_for_reply:
                    #         self.sent_action_index = self.action_index - 1
                    #         self.socket.send_pyobj(aligned_obs)
                    #         self.waiting_for_reply = True # LOCK the socket (prevent double sends)
                    #         print("Request sent.")

                    #     # --- STEP B: CHECK IF WE CAN RECEIVE ---
                    #     if self.waiting_for_reply:
                    #         try:
                    #             # Check for data without blocking
                    #             actions = self.socket.recv_pyobj(flags=zmq.NOBLOCK)
                                
                    #             # If we get here, we successfully received data
                    #             self.waiting_for_reply = False # UNLOCK the socket
                    #             print(f"Received actions. Obs sent @ {self.sent_action_index}, current action index {self.action_index}")
                    #             # Update your logic
                    #             self.action_index -= self.sent_action_index
                    #             self.sent_action_index = self.action_index
                                
                                
                    #         except zmq.Again:
                    #             # No message ready yet. This is normal.
                    #             # We just continue the loop and try again next tick.
                    #             actions = None
                                                            
                    # #5. Process and store actions
                    if actions is not None and actions.shape[-1] == 14:
                        # print("Received actions from model.")

                        # actions shape: (B, action_horizon, 14)
                        # Split into: joint_pos_L (6), gripper_pos_L (1), joint_pos_R (6), gripper_pos_R (1)
                        action_horizon, action_dim = actions.shape
                        
                        # Store predictions (using first batch item)
                        
                        self.predicted_actions = {
                            "joint_pos_L": actions[:, :6],
                            "gripper_pos_L": actions[:, 6],
                            "joint_pos_R": actions[:, 7:13],
                            "gripper_pos_R": actions[:, 13] if action_dim > 13 else 0.0
                        }
                        # print(f"Stored {action_horizon} actions for execution: \n {self.predicted_actions}")
                    
                    # #6. Send current action to robot if available
                    assert (self.action_index < len(self.predicted_actions["joint_pos_L"]) and 
                        len(self.predicted_actions["joint_pos_L"]) > 0), "No predicted actions to send to robot!"
                    
                    
                    command = {
                        "joint_pos_L": self.predicted_actions["joint_pos_L"][self.action_index].astype(np.float32),
                        "gripper_pos_L": np.float32(self.predicted_actions["gripper_pos_L"][self.action_index]),
                        "joint_pos_R": self.predicted_actions["joint_pos_R"][self.action_index].astype(np.float32), 
                        "gripper_pos_R": np.float32(self.predicted_actions["gripper_pos_R"][self.action_index])
                    }

                    if prev_action is not None:
                        # Apply EMA smoothing
                        for key in command.keys():
                            command[key] = ((1-ema_alpha) * prev_action[key] + (ema_alpha) * command[key]).astype(command[key].dtype)
                    prev_action = command
                    
                    

                    if self.execute:
                        command_exec.append(command)
                        try:
                            self.robot_joint_queue.put(command)
                            print(f"Sent action {self.action_index}/{len(self.predicted_actions['joint_pos_L'])}")
                            self.action_index += 1
                        except Exception as e:
                            print(f"Failed to send command to robot: {e}")

                    #7. Handle keystrokes
                    press_events = key_counter.get_press_events()
                    for key_stroke in press_events:
                        if key_stroke == KeyCode(char='q'):
                            running = False
                        elif key_stroke == KeyCode(char='r'):
                            self.start_recording()
                            key_counter.clear()
                            self.is_recording = True
                            print("*"*10, 'Recording!', "*"*10)
                        elif key_stroke == KeyCode(char='e'):
                            self.stop_recording()
                            key_counter.clear()
                            self.is_recording = False
                            print("*"*10, 'Stopped.', "*"*10)

                        elif key_stroke == Key.backspace:
                            key_counter.clear()
                            is_recording = False
                    
                    #8. Enforce loop rate
                    remaining = (1.0 / self.frequency) - (time.monotonic() - t)
                    if remaining > 0:
                        time.sleep(remaining)

                    # Write stitched camera frame if recording
                    if self.is_recording:
                        stitched = self.get_stitched_camera_frame()
                        if stitched is not None:
                            self.ensure_video_writer(stitched.shape[1], stitched.shape[0])
                            self.video_writer.write(stitched)
                    print(f"Loop Frequency: {1.0 / (time.monotonic() - t):.3f} Hz")
                    
            finally:
                self.finish()
                sample_indices = np.arange(len(command_exec))
                allLeft_joints = np.array([cmd["joint_pos_L"] for cmd in command_exec])
                allRight_joints = np.array([cmd["joint_pos_R"] for cmd in command_exec])
                allLeftGripper = np.array([cmd["gripper_pos_L"] for cmd in command_exec]).reshape(-1,1)
                allRightGripper = np.array([cmd["gripper_pos_R"] for cmd in command_exec]).reshape(-1,1)
                concatAll = np.hstack((allLeft_joints, allLeftGripper, allRight_joints, allRightGripper))
                fig, axes = plt.subplots(4, 4, figsize=(20, 16))
                fig.suptitle(f'Episode - Model Plot Per Dimension', fontsize=16)
        
                # Plot all dimensions in one loop
                for i in range(14):
                    row = i // 4
                    col = i % 4
                    
                    if row < 4 and col < 4:  # Make sure we don't exceed subplot grid
                        ax = axes[row, col]
                        
                        # Determine robot and joint type
                        if i < 7:  # Robot 0 territory (0-6)
                            robot_name = "Robot 0"
                            if i == 6:
                                joint_type = "Gripper"
                            else:
                                joint_type = f"Joint {i}"
                        else:  # Robot 1 territory (7-13)
                            robot_name = "Robot 1"
                            if i == 13:
                                joint_type = "Gripper"
                            else:
                                joint_type = f"Joint {i-7}"
                        
                        ax.plot(sample_indices, concatAll[:, i], label=f'Pred', alpha=0.7, linestyle='--', linewidth=2)
                        ax.set_title(f'{robot_name} - {joint_type}', fontsize=12)
                        ax.legend()
                        ax.grid(True, alpha=0.3)
                        ax.set_xlabel('Sample Index')
                        ax.set_ylabel('Action Value')
                plt.show()

    def start_recording(self):
        self.cmd_q_robot.put({'type': 'START_REC', 'ep_idx': self.ep_idx})
        for serial, cmd_q in self.cmd_q_cameras.items():
            cmd_q.put({'type': 'START_REC', 'ep_idx': self.ep_idx})
        self.execute = True
        # reset video path per episode
        self.video_path = pathlib.Path.cwd().joinpath(f"camera_stitched_ep{self.ep_idx}_{int(time.time())}.mp4")
        self.is_recording = True
    

    def stop_recording(self):
        self.cmd_q_robot.put({'type': 'STOP_REC', 'ep_idx': self.ep_idx})
        for serial, cmd_q in self.cmd_q_cameras.items():
            cmd_q.put({'type': 'STOP_REC', 'ep_idx': self.ep_idx})
        self.execute = False
        self.is_recording = False
        # release writer
        if self.video_writer is not None:
            try:
                self.video_writer.release()
            except Exception:
                pass
            self.video_writer = None

    def finish(self):
        self.cmd_q_robot.put({'type': 'EXIT', 'ep_idx': self.ep_idx})
        for serial, cmd_q in self.cmd_q_cameras.items():
            cmd_q.put({'type': 'EXIT', 'ep_idx': self.ep_idx})
        self.robot_proc.join()
        for serial, camera_proc in self.camera_procs.items():
            camera_proc.join()
        self.execute = False
        # Cleanup shared memory manager
        self.shm_manager.shutdown()
        # Release video writer if open
        if self.video_writer is not None:
            try:
                self.video_writer.release()
            except Exception:
                pass
            self.video_writer = None

    def ensure_video_writer(self, width, height):
        if self.video_writer is None:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            self.video_writer = cv2.VideoWriter(str(self.video_path), fourcc, self.video_fps, (width, height))

    def get_stitched_camera_frame(self):
        try:
            # Get latest frames from each camera
            frames = []
            for serial, buffer in self.camera_obs_buffers.items():
                data = buffer.get_last_k(1)
                # color shape (H,W,3), uint8
                color = data["color"][0]
                frames.append(color)

            if len(frames) == 0:
                return None

            # Determine layout: try 2 rows if many cameras
            n = len(frames)
            cols = int(np.ceil(np.sqrt(n)))
            rows = int(np.ceil(n / cols))

            # Resize all frames to smallest height
            target_h = min(f.shape[0] for f in frames)
            target_w = min(f.shape[1] for f in frames)
            resized = [cv2.resize(f, (target_w, target_h)) for f in frames]

            # Pad list to fill grid
            pad = rows * cols - n
            if pad > 0:
                black = np.zeros((target_h, target_w, 3), dtype=np.uint8)
                resized.extend([black] * pad)

            # Build grid
            rows_list = []
            for r in range(rows):
                row_imgs = resized[r*cols:(r+1)*cols]
                row_cat = np.concatenate(row_imgs, axis=1)
                rows_list.append(row_cat)
            grid = np.concatenate(rows_list, axis=0)

            # Convert to BGR for VideoWriter if needed (already BGR)
            return grid
        except Exception as e:
            print(f"Failed to stitch camera frames: {e}")
            return None

    def get_aligned_observations(self):
        """
        Get aligned robot and camera observations from ring buffers.
        Returns dict with robot and camera observations, aligned by timestamp.
        """
        try:
            # Get latest robot observation
            
            # Get camera observations and find closest timestamps

            k = self.camera_obs_horizon + 2  # extra buffer
            self.last_camera_data = {}
            
            for serial, buffer in self.camera_obs_buffers.items():
                try:
                    # Get latest k camera observations
                    camera_obs = buffer.get_last_k(k)
                    self.last_camera_data[serial] = camera_obs
                    
                except Exception as e:
                    print(f"Failed to get camera {serial} observation: {e}")
                    return None
            

            self.last_robot_data = self.robot_obs_buffer.get_all()

            align_camera_serial = None
            running_best_error = np.inf
            camera_serials = list(self.camera_obs_buffers.keys())
            
            for camera_serial in camera_serials:
                this_error = 0
                this_timestamp = self.last_camera_data[camera_serial]["timestamp"][-1]
                for other_camera_serial in camera_serials:
                    if other_camera_serial == camera_serial:
                        continue
                    other_timestep_idx = -1
                    while True:
                        if (
                            self.last_camera_data[other_camera_serial]["timestamp"][
                                other_timestep_idx
                            ]
                            < this_timestamp
                        ):
                            this_error += (
                                this_timestamp
                                - self.last_camera_data[other_camera_serial]["timestamp"][
                                    other_timestep_idx
                                ]
                            )
                            break
                        other_timestep_idx -= 1
                if align_camera_serial is None or this_error < running_best_error:
                    running_best_error = this_error
                    align_camera_serial = camera_serial

            last_timestamp = self.last_camera_data[align_camera_serial]["timestamp"][-1]
            dt = 1 / self.frequency

            # align camera obs timestamps
            camera_obs_timestamps = last_timestamp - (
                np.arange(self.camera_obs_horizon)[::-1]
                * self.camera_down_sample_steps
                * dt
            )

            camera_obs = dict()
            for camera_idx, (serial, value) in enumerate(self.last_camera_data.items()):
                this_timestamps = value["timestamp"]
                this_idxs = list()
                for t in camera_obs_timestamps:
                    nn_idx = np.argmin(np.abs(this_timestamps - t))
                    this_idxs.append(nn_idx)
                # remap key
                camera_obs[f"camera{camera_idx}_rgb"] = np.moveaxis(value["color"][this_idxs], -1, 1).astype(np.float32) / 255.

            # obs_data to return (it only includes camera data at this stage)
            obs_data = dict(camera_obs)

            # include camera timesteps
            # obs_data["timestamp"] = camera_obs_timestamps

            robot_obs_timestamps = last_timestamp - (
                np.arange(self.robot_obs_horizon)[::-1] * self.robot_down_sample_steps * dt
            )

            for key, value in self.last_robot_data.items():
                if key != "timestamp":
                    interp = si.interp1d(
                        self.last_robot_data["timestamp"], value, axis=0, bounds_error=False, fill_value=(value[0], value[-1])
                    )
                    self.last_robot_data[key] = interp(robot_obs_timestamps)
            
            obs_data['robots_joint_action'] = preprocess_robot_actions(self.last_robot_data, timestamps_key=None)['action']
            
                
            return obs_data
            
        except Exception as e:
            print(f"Failed to get robot observation: {e}")
            return None


    def get_connected_devices_serial(self):
        serials = list()
        for d in rs.context().devices:
            if d.get_info(rs.camera_info.name).lower() != 'platform camera':
                serial = d.get_info(rs.camera_info.serial_number)
                product_line = d.get_info(rs.camera_info.product_line)
                if product_line == 'D400':
                    # only works with D400 series
                    serials.append(serial)
        serials = sorted(serials)
        return serials


c = Conductor()
c.loop()