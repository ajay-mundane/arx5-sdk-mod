from pynput.keyboard import Key, KeyCode, Listener
from collections import defaultdict
from threading import Lock
import multiprocessing as mp
from robotcontrol import RobotControlProcess
from cameracapture import CameraCaptureProcess
import pyrealsense2 as rs
from helper import ReplayBuffer, match_observations_to_actions, align_camera_timestamps, preprocess_robot_actions
import numpy as np
import pathlib
from omegaconf import OmegaConf
from tqdm import tqdm

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

    def __init__(self, save_dir=None):
        
        # Create communication channels
        self.cmd_q_robot = mp.Queue()
        self.cmd_q_cameras = {}

        self.data_q_robot = mp.Queue()
        self.data_q_cameras = {}

        # Hardware Config - load from YAML
        config_path = pathlib.Path(__file__).resolve().parent.joinpath('robotcontrol.yaml')
        rcfg = OmegaConf.load(str(config_path))
        leader_cfg = (rcfg.leader.model, rcfg.leader.left, rcfg.leader.right)
        follower_cfg = (rcfg.follower.model, rcfg.follower.left, rcfg.follower.right)
        dof = int(rcfg.dof)
        friction_cfg = OmegaConf.to_container(rcfg.friction, resolve=True)

        # Camera config
        camera_config = {
            "resolution": (640, 480),
            "capture_fps": 30,
            "enable_color": True,
            "enable_depth": False
        }

        # Start Robot Process
        robot_cfg = {"leader": leader_cfg, "follower": follower_cfg, "dof": dof, "friction": friction_cfg}
        self.robot_proc = RobotControlProcess(self.cmd_q_robot, self.data_q_robot, robot_cfg)
        self.robot_proc.start()

        # Start Cameras (as discussed before)
        self.camera_procs = {} # let's loop this for all cameras
        serials = self.get_connected_devices_serial()
        for serial in serials:
            cmd_q_camera = mp.Queue()
            data_q_camera = mp.Queue()
            self.cmd_q_cameras[serial] = cmd_q_camera
            self.data_q_cameras[serial] = data_q_camera
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
            new_camera_proc = CameraCaptureProcess(cmd_q_camera, data_q_camera, camera_config, serial)
            self.camera_procs[serial] = new_camera_proc
            new_camera_proc.start()

        self.ep_idx = 0
        self.save_to_disk = False
        if save_dir:
            self.save_to_disk = True
            output_dir = pathlib.Path(save_dir)
            assert output_dir.parent.is_dir()

            zarr_path = str(output_dir.joinpath('replay_buffer.zarr').absolute())
            self.replay_buffer = ReplayBuffer.create_from_path(
                zarr_path=zarr_path, mode='a')
            
            self.ep_idx = self.replay_buffer.n_episodes


    def loop(self):
        with KeystrokeCounter() as key_counter:
            try:
                running = True
                while running:

                    press_events = key_counter.get_press_events()
                    for key_stroke in press_events:
                        if key_stroke == KeyCode(char='q'):
                            running = False
                        elif key_stroke == KeyCode(char='r'):
                            self.start_recording()
                            key_counter.clear()
                            is_recording = True
                            print("*"*10, 'Recording!', "*"*10)
                        elif key_stroke == KeyCode(char='e'):
                            self.stop_recording()
                            key_counter.clear()
                            is_recording = False
                            print("*"*10, 'Stopped.', "*"*10)

                            self.receive_and_save_data()

                        elif key_stroke == Key.backspace:
                            key_counter.clear()
                            is_recording = False
                    
            finally:
                self.finish()

    def start_recording(self):
        self.cmd_q_robot.put({'type': 'START_REC', 'ep_idx': self.ep_idx})
        for serial, cmd_q in self.cmd_q_cameras.items():
            cmd_q.put({'type': 'START_REC', 'ep_idx': self.ep_idx})
    

    def stop_recording(self):
        self.cmd_q_robot.put({'type': 'STOP_REC', 'ep_idx': self.ep_idx})
        for serial, cmd_q in self.cmd_q_cameras.items():
            cmd_q.put({'type': 'STOP_REC', 'ep_idx': self.ep_idx})

    def finish(self):
        self.cmd_q_robot.put({'type': 'EXIT', 'ep_idx': self.ep_idx})
        for serial, cmd_q in self.cmd_q_cameras.items():
            cmd_q.put({'type': 'EXIT', 'ep_idx': self.ep_idx})
        self.robot_proc.join()
        for serial, camera_proc in self.camera_procs.items():
            camera_proc.join()


    def _receive_chunked_data(self, data_queue, source_name):
        """Receive and reassemble chunked data from a process"""
        chunks = []
        total_chunks = None
        
        # Receive all chunks
        while True:
            chunk_msg = data_queue.get()
            chunks.append(chunk_msg)
            
            if total_chunks is None:
                total_chunks = chunk_msg['total_chunks']
                print(f"Expecting {total_chunks} chunks from {source_name}")
            
            if len(chunks) >= total_chunks:
                break
        
        # Sort chunks by chunk_id to ensure correct order
        chunks.sort(key=lambda x: x['chunk_id'])
        
        # Reassemble data
        if total_chunks == 1 and len(chunks[0]['data']['cam_timestamps'] if 'cam_timestamps' in chunks[0]['data'] else chunks[0]['data']['action_timestamps']) == 0:
            # Handle empty data case
            return chunks[0]['data']
        
        assembled_data = {}
        for key in chunks[0]['data'].keys():
            # Concatenate all chunks for this key
            key_data = [chunk['data'][key] for chunk in chunks]
            assembled_data[key] = np.concatenate(key_data, axis=0)
        
        return assembled_data
    
    def receive_and_save_data(self):
        # 1. Get data from queues after end recording (now with chunking)
        print("Receiving robot data...")
        robot_action_data = self._receive_chunked_data(self.data_q_robot, "robot")
        
        # Collect all camera data
        cameras_data = {}
        for serial, data_q in self.data_q_cameras.items():
            print(f"Receiving camera {serial} data...")
            camera_data = self._receive_chunked_data(data_q, f"camera-{serial}")
            cameras_data[serial] = camera_data
            printer={key: value.shape for key, value in camera_data.items()}
            print(f"cam serial {serial} {printer}")
        
        print(f"Received data: robot {robot_action_data['joint_pos_R'].shape}")
        
        # 2. Align camera timestamps using camera with least frames as baseline
        print("Aligning camera timestamps...")
        aligned_camera_data = align_camera_timestamps(cameras_data)
        
        # 3. Preprocess robot actions - combine into single action array
        print("Preprocessing robot actions...")
        processed_robot_data = preprocess_robot_actions(robot_action_data)
        
        # 4. Match robot actions to aligned camera observations using baseline timestamps
        print("Matching robot actions to camera observations...")
        action_idxs, mask = match_observations_to_actions(
            processed_robot_data['action_timestamps'],
            aligned_camera_data['cam_timestamps']
        )
        
        # Apply alignment to robot data
        for key, value in processed_robot_data.items():
            processed_robot_data[key] = value[action_idxs]
        
        # Apply alignment to camera data
        for key, value in aligned_camera_data.items():
            assert len(value) == len(mask), f"Length mismatch for camera data key {key}: len(value)={len(value)}, len(mask)={len(mask)}"
            aligned_camera_data[key] = value[mask]
        
        # Verify alignment
        n_acts = len(processed_robot_data['action_timestamps'])
        n_obs = len(aligned_camera_data['cam_timestamps'])
        assert n_acts == n_obs, f"Mismatch! Robot: {n_acts}, Cam: {n_obs}"
        
        print(f"Final aligned data: {n_acts} timesteps")
        print(f"Action shape: {processed_robot_data['action'].shape}")
        
        # 5. Save data to replay buffer
        print("Saving episode...")
        if self.save_to_disk:
            episode_data = processed_robot_data | aligned_camera_data
            self.replay_buffer.add_episode(episode_data, compressors='disk')
            episode_id = self.replay_buffer.n_episodes - 1
            self.ep_idx = self.replay_buffer.n_episodes
            
            print(f'Episode {episode_id} saved! Approx time: {n_obs/30:.2f} sec')
            
            # Print summary of saved data
            for key, value in episode_data.items():
                if hasattr(value, 'shape'):
                    print(f"  {key}: {value.shape}")
                else:
                    print(f"  {key}: {type(value)}")



    

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


c = Conductor(save_dir="./data2") # "./data"
c.loop()