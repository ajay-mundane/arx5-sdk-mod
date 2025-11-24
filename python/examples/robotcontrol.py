from collections import defaultdict
import multiprocessing as mp
import time
import numpy as np
import pickle
import sys
import os
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)
os.chdir(ROOT_DIR)
import arx5_interface as arx5
from helper import NumpyAccumulator

MAX_TORQUE = 1.5 / 2 - 1e-2

K_COULOMB = 0.11  # Nm (Assists your push)
K_VISCOUS = 0.05 # Nm/(rad/s)

VEL_DEADBAND = 0.02 # rad/s (Ignore small accidental movements)

def sign(x):
    return 1.0 if x > 0 else -1.0 if x < 0 else 0.0
def friction_compensation(current_vel):
    # vel: joint velocity (rad/s)
    if abs(current_vel) > VEL_DEADBAND:
        return (K_COULOMB * sign(current_vel)) + (K_VISCOUS * current_vel)
    else:
        return 0.0

# --- The Robot Process Class ---
class RobotControlProcess(mp.Process):
    def __init__(self, cmd_queue, data_queue, config):
        super().__init__(name="Robot-Worker")
        self.cmd_queue = cmd_queue       # To receive high-level commands
        self.data_queue = data_queue   # To send latest state for UI/Monitoring
        self.config = config
        self.is_recording = False
        self.loop = True
        self.action_buffer = {
            "joint_pos_L": NumpyAccumulator(shape_suffix=(config["dof"],)),
            "gripper_pos_L": NumpyAccumulator(),
            "joint_pos_R": NumpyAccumulator(shape_suffix=(config["dof"],)),
            "gripper_pos_R": NumpyAccumulator(),
            "action_timestamps": NumpyAccumulator(dtype=np.float64)
        }

    def run(self):
        try:
            # Initialize hardware IN THIS PROCESS
            l_model, l_can1, l_can2 = self.config['leader']
            f_model, f_can1, f_can2 = self.config['follower']

            leader_L = arx5.Arx5JointController(l_model, l_can1)
            leader_R = arx5.Arx5JointController(l_model, l_can2)
            follower_L = arx5.Arx5JointController(f_model, f_can1)
            follower_R = arx5.Arx5JointController(f_model, f_can2)
            
            dt = leader_L.get_controller_config().controller_dt
            
            # Reset and setup logic here...
            robot_config_leader = leader_L.get_robot_config()
            self.dof = robot_config_leader.joint_dof

            leader_L.reset_to_home()
            follower_L.reset_to_home()
            leader_R.reset_to_home()
            follower_R.reset_to_home()

            gain = arx5.Gain(self.dof)
            gain.kd()[:] = 0.01
            gain.gripper_kp = 0.0
            gain.gripper_kd = 0.0
            leader_L.set_gain(gain)

            gain = arx5.Gain(self.dof)
            gain.kd()[:] = 0.01
            gain.gripper_kp = 0.0
            gain.gripper_kd = 0.0
            leader_R.set_gain(gain)

            self.t_start = time.monotonic()
            self.iter_idx = 0
            self.dt = 1/400.0
            while self.loop:
                now_mono = time.monotonic()
                now_epoch = time.time()

                # schedule the timestamp for this cycle
                t_cycle_end = self.t_start + (self.iter_idx + 1) * self.dt
                
                # 1. Handle Commands
                self._handle_commands()

                # 2. Teleoperation Logic
                action_data = self._run_teleop_step(leader_L, leader_R, follower_L, follower_R)
                # convert scheduled monotonic -> epoch
                timestamp = t_cycle_end - now_mono + now_epoch

                if self.is_recording:
                    self._put_data(action_data, timestamp)

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
            leader_L.reset_to_home()
            leader_R.reset_to_home()
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
                self._send_data()
                # Send confirmation back to Conductor after saving
                

    def _run_teleop_step(self, l_L, l_R, f_L, f_R):
        
        # Read Leader States
        l_state_L = l_L.get_joint_state() # leader state left
        l_state_R = l_R.get_joint_state() # leader state right

        # Compute & Write Leader Commands (Friction Comp)
        tau_L = friction_compensation(l_state_L.gripper_vel)
        tau_R = friction_compensation(l_state_R.gripper_vel)
        
        joint_cmd = arx5.JointState(self.dof)
        joint_cmd.gripper_torque = tau_L
        l_L.set_joint_cmd(joint_cmd)

        joint_cmd = arx5.JointState(self.dof)
        joint_cmd.gripper_torque = tau_R
        l_R.set_joint_cmd(joint_cmd)

        # Write Follower Commands (Position Tracking)
        f_cmd_L = arx5.JointState(self.dof)
        f_cmd_L.pos()[:] = l_state_L.pos()
        f_cmd_L.gripper_pos = l_state_L.gripper_pos
        
        
        f_cmd_R = arx5.JointState(self.dof)
        f_cmd_R.pos()[:] = l_state_R.pos()
        f_cmd_R.gripper_pos = l_state_R.gripper_pos


        f_L.set_joint_cmd(f_cmd_L)
        f_R.set_joint_cmd(f_cmd_R)

        return {
            "joint_pos_L": f_cmd_L.pos().copy(),
            "gripper_pos_L": f_cmd_L.gripper_pos,
            "joint_pos_R": f_cmd_R.pos().copy(),
            "gripper_pos_R": f_cmd_R.gripper_pos,
        }


        # Send latest state to Conductor for UI update (throttled)
        # if self.state_queue.empty(): self.state_queue.put(pickle.dumps(data_packet))

    def _put_data(self, action_data, timestamp):
        for x,y in action_data.items():
            self.action_buffer[x].append(y)
        self.action_buffer['action_timestamps'].append(timestamp)
    
    def _send_data(self):
        send = {k:v.data for k,v in self.action_buffer.items()}
        self.data_queue.put(send)
        for k, v in self.action_buffer.items():
            v.reset()
