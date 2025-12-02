from collections import defaultdict
import multiprocessing as mp
import time
import numpy as np
import pickle
import sys
import os
from tqdm import tqdm
from queue import Empty
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)
os.chdir(ROOT_DIR)
import arx5_interface as arx5
from helper import NumpyAccumulator

MAX_TORQUE = 1.5 / 2 - 1e-2
K_COULOMB = 0.11  # default fallback
K_VISCOUS = 0.05  # default fallback
VEL_DEADBAND = 0.02

def sign(x):
    return 1.0 if x > 0 else -1.0 if x < 0 else 0.0

# --- The Robot Process Class ---
class RobotControlProcess(mp.Process):
    def __init__(self, cmd_queue, config, obs_buffer=None, input_queue=None):
        super().__init__(name="Robot-Worker")
        self.cmd_queue = cmd_queue       # To receive high-level commands
        self.config = config
        self.obs_buffer = obs_buffer     # Shared memory ring buffer for observations
        self.input_queue = input_queue   # Shared memory queue for predicted actions post inference
        self.is_recording = False
        self.loop = True

    def run(self):
        try:
            # Initialize hardware IN THIS PROCESS
            f_model, f_can1, f_can2 = self.config['follower']

            follower_L = arx5.Arx5JointController(f_model, f_can1)
            follower_R = arx5.Arx5JointController(f_model, f_can2)
            
            
            # Reset and setup logic here...
            robot_config_leader = follower_L.get_robot_config()
            self.dof = robot_config_leader.joint_dof

            # Set gains for smoother movement
            gain = arx5.Gain(self.dof)
            gain.kd()[:] = 0.01  # Lower damping for smoother tracking
            gain.gripper_kp = 0.0
            gain.gripper_kd = 0.0
            follower_L.set_gain(gain)
            
            gain_R = arx5.Gain(self.dof)
            gain_R.kd()[:] = 0.01
            gain_R.gripper_kp = 0.0
            gain_R.gripper_kd = 0.0
            follower_R.set_gain(gain_R)

            follower_L.reset_to_home()
            follower_R.reset_to_home()

            self.t_start = time.monotonic()
            self.iter_idx = 0
            self.dt = 1/100.0
            while self.loop:
                now_mono = time.monotonic()
                now_epoch = time.time()

                # schedule the timestamp for this cycle
                t_cycle_end = self.t_start + (self.iter_idx + 1) * self.dt
                
                # 1. Handle Commands
                self._handle_commands()

                # 1.5. Check for predicted actions from shared memory queue
                predicted_commands = None
                if self.input_queue is not None:
                    try:
                        # process at most 1 command per cycle to maintain frequency
                        commands = self.input_queue.get_k(1)
                        n_cmd = len(commands["joint_pos_L"]) if "joint_pos_L" in commands else 0
                        if n_cmd > 0:
                            predicted_commands = commands
                    except Empty:
                        predicted_commands = None
                    except Exception as e:
                        print(f"[Robot-Worker] Error getting predicted commands: {e}")
                        predicted_commands = None

                # 2. Teleoperation Logic
                action_data = self._run_teleop_step(follower_L, follower_R, predicted_commands=predicted_commands)
                # convert scheduled monotonic -> epoch
                timestamp = t_cycle_end - now_mono + now_epoch

                # 3. Put observations in ring buffer (always, not just when recording)
                if self.obs_buffer is not None:
                    obs_data = {
                        "joint_pos_L": action_data["joint_pos_L"],
                        "gripper_pos_L": action_data["gripper_pos_L"],
                        "joint_pos_R": action_data["joint_pos_R"],
                        "gripper_pos_R": action_data["gripper_pos_R"],
                        "timestamp": timestamp
                    }
                    try:
                        self.obs_buffer.put(obs_data, wait=False)
                    except Exception as e:
                        print(f"[Robot-Worker] Failed to put obs in buffer: {e}")


                # 3. Enforce Loop Rate
                remaining = t_cycle_end - time.monotonic()
                if remaining > 0:
                    time.sleep(remaining)

                duration = time.time() - now_epoch
                frequency = np.round(1 / duration, 1)
                # print(f"FPS: {frequency}")

                self.iter_idx += 1

        except Exception as e:
            print(f"[Robot-Worker] Critical Error: {e}")
        finally:
            print("[Robot-Worker] Shutting down arms...")
            follower_L.reset_to_home()
            follower_R.reset_to_home()

    def _handle_commands(self):
        if not self.cmd_queue.empty():
            cmd = self.cmd_queue.get()
            
            if cmd['type'] == 'EXIT': self.loop = False

            elif cmd['type'] == 'START_REC':
                self.is_recording = True
                self.episode_idx = cmd['ep_idx']

            elif cmd['type'] == 'STOP_REC':
                self.is_recording = False
                
                # Send confirmation back to Conductor after saving
                

    def _run_teleop_step(self, f_L, f_R, predicted_commands=None):
        
        # Write Follower Commands (Position Tracking)
        f_cmd_L = arx5.JointState(self.dof)
        f_cmd_R = arx5.JointState(self.dof)
        
        if predicted_commands is not None:
            # Use predicted actions if available
            f_cmd_L.pos()[:] = predicted_commands["joint_pos_L"]
            f_cmd_L.gripper_pos = predicted_commands["gripper_pos_L"]
            
            f_cmd_R.pos()[:] = predicted_commands["joint_pos_R"]
            f_cmd_R.gripper_pos = predicted_commands["gripper_pos_R"]

            f_L.set_joint_cmd(f_cmd_L)
            f_R.set_joint_cmd(f_cmd_R)

            return {
                "joint_pos_L": f_cmd_L.pos().copy(),
                "gripper_pos_L": f_cmd_L.gripper_pos,
                "joint_pos_R": f_cmd_R.pos().copy(),
                "gripper_pos_R": f_cmd_R.gripper_pos,
            }
            
        else:
            current_state_L = f_L.get_joint_state()
            current_state_R = f_R.get_joint_state()

            return {
                "joint_pos_L": current_state_L.pos().copy(),
                "gripper_pos_L": current_state_L.gripper_pos,
                "joint_pos_R": current_state_R.pos().copy(),
                "gripper_pos_R": current_state_R.gripper_pos,
            }

        

        


        # Send latest state to Conductor for UI update (throttled)
        # if self.state_queue.empty(): self.state_queue.put(pickle.dumps(data_packet))
