from pynput.keyboard import Key, KeyCode, Listener
from collections import defaultdict
from threading import Lock
import multiprocessing as mp
from robotcontrol import RobotControlProcess
from cameracapture import CameraCaptureProcess
import pyrealsense2 as rs


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
        
        # Create communication channels
        self.cmd_q_robot = mp.Queue()
        self.cmd_q_camera = mp.Queue()

        self.data_q_robot = mp.Queue()
        self.data_q_camera = mp.Queue()

        # Hardware Config
        leader_cfg = ('X5', 'can0', 'can5') # Model, Left Interface, Right Interface
        follower_cfg = ('X5', 'can3', 'can1')

        # Camera config
        camera_config = {
            "resolution": (640, 480),
            "capture_fps": 30,
            "enable_color": True,
            "enable_depth": True
        }

        # Start Robot Process
        self.robot_proc = RobotControlProcess(self.cmd_q_robot, self.data_q_robot, {"leader":leader_cfg, "follower":follower_cfg})
        self.robot_proc.start()

        # Start Cameras (as discussed before)
        # self.camera_procs = [] let's loop this for all cameras
        serial = self.get_connected_devices_serial()[0]
        print(serial)
        self.camera_proc = CameraCaptureProcess(self.cmd_q_camera, self.data_q_camera, camera_config, serial)
        self.camera_proc.start()

        self.ep_idx = 0


    def loop(self):
        print("HERE"*20)
        with KeystrokeCounter() as key_counter:
            try:
                running = True
                while running:

                    press_events = key_counter.get_press_events()
                    for key_stroke in press_events:
                        # if key_stroke != None:
                        #     print(key_stroke)
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

                            self.ep_idx += 1
                        elif key_stroke == Key.backspace:
                            key_counter.clear()
                            is_recording = False
                    
            finally:
                self.finish()

    def start_recording(self):
        self.cmd_q_robot.put({'type': 'START_REC', 'ep_idx': self.ep_idx})
        self.cmd_q_camera.put({'type': 'START_REC', 'ep_idx': self.ep_idx})
    

    def stop_recording(self):
        self.cmd_q_robot.put({'type': 'STOP_REC', 'ep_idx': self.ep_idx})
        self.cmd_q_camera.put({'type': 'STOP_REC', 'ep_idx': self.ep_idx})

    def finish(self):
        self.cmd_q_robot.put({'type': 'EXIT', 'ep_idx': self.ep_idx})
        self.cmd_q_camera.put({'type': 'EXIT', 'ep_idx': self.ep_idx})
        self.robot_proc.join()
        self.camera_proc.join()


    def receive_and_save_data(self):
        # 1. get data from queues after end recording
        robot_action_data = self.data_q_robot.get()
        camera_obs_data = self.data_q_camera.get()
        print("Received data")

        # 2. match each robot action data to a camera obs using timestamps - testing purposes its fine



        # NOTE: at this point, we will have {joint_pos: np.array(ep_len, D), gripper_pos: np.array(ep_len, D), action_timestamps: np.array(ep_len), cameraL: np.array(ep_len), cameraR: np.array(ep_len), cameraHead: np.array(ep_len)}
        # 3. save data - replay buffer stuff
        print(robot_action_data.keys())
        print(robot_action_data['gripper_pos_R'].max())
        print("Saving part")



    

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