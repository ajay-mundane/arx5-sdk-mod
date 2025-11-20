from pynput.keyboard import Key, KeyCode, Listener
from collections import defaultdict
from threading import Lock
import multiprocessing as mp
from robotcontrol import RobotControlProcess

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

        self.state_q = mp.Queue()

        # Hardware Config
        leader_cfg = ('X5', 'can0', 'can2') # Model, Left Interface, Right Interface
        follower_cfg = ('X5', 'can3', 'can1')

        # Start Robot Process
        self.robot_proc = RobotControlProcess(self.cmd_q_robot, self.state_q, {"leader":leader_cfg, "follower":follower_cfg})
        self.robot_proc.start()

        # Start Cameras (as discussed before)
        # camera_proc.start()...

        self.ep_idx = 0


    def loop(self):
        print("HERE"*20)
        with KeystrokeCounter() as key_counter:
            try:
                running = True
                while running:
                    # 1. Get latest robot state for visualization
                    if not self.state_q.empty():
                        robot_state = self.state_q.get()
                        # Update your CV2 Window text with robot_state['left_gripper']...

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
                            print('Recording!')
                        elif key_stroke == KeyCode(char='e'):
                            self.stop_recording()
                            key_counter.clear()
                            is_recording = False
                            print('Stopped.')
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
        # self.camer_proc.join()


c = Conductor()
c.loop()