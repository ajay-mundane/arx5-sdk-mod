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

MAX_TORQUE = 0.75

def friction_compensation(vel):
    # vel: joint velocity (rad/s)
    k = 0.45             # your viscous coefficient
    assist = 0.05        # constant help torque
    v_dead = 0.003       # below this, treat as “not moving”

    if abs(vel) < v_dead:
        tau = 0.0                               # no movement → no torque
    else:
        direction = np.sign(vel)
        tau = k * vel + assist * direction      # add small constant help
    return float(np.clip(tau, -MAX_TORQUE, MAX_TORQUE))

# --- The Robot Process Class ---
class RobotControlProcess(mp.Process):
    def __init__(self, cmd_queue, state_queue, config):
        super().__init__(name="Robot-Worker")
        self.cmd_queue = cmd_queue       # To receive high-level commands
        self.state_queue = state_queue   # To send latest state for UI/Monitoring
        self.config = config
        self.is_recording = False
        self.recorded_data = []
        self.loop = True

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
            
            while self.loop:
                loop_start = time.time()
                
                # 1. Handle Commands
                self._handle_commands()

                # 2. Teleoperation Logic
                self._run_teleop_step(leader_L, leader_R, follower_L, follower_R)

                # 3. Enforce Loop Rate
                elapsed = time.time() - loop_start
                if dt - elapsed > 0:
                    time.sleep(dt - elapsed)

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
                self.recorded_data = [] 

            elif cmd['type'] == 'STOP_REC':
                self.is_recording = False
                self._save_data(self.episode_idx)
                # Send confirmation back to Conductor after saving
                self.cmd_queue.put({'type': 'STOPPED', 'ep_idx': self.episode_idx})

    def _run_teleop_step(self, l_L, l_R, f_L, f_R):
        current_timestamp = time.time()
        
        # Read Leader States
        l_state_L = l_L.get_joint_state()
        l_state_R = l_R.get_joint_state()

        # Compute & Write Leader Commands (Friction Comp)
        # tau_L = friction_compensation(l_state_L.gripper_vel)
        # tau_R = friction_compensation(l_state_R.gripper_vel)
        
        # joint_cmd = arx5.JointState(self.dof)
        # joint_cmd.gripper_torque = tau_L
        # l_L.set_joint_cmd(joint_cmd)

        # joint_cmd = arx5.JointState(self.dof)
        # joint_cmd.gripper_torque = tau_R
        # l_R.set_joint_cmd(joint_cmd)

        # Write Follower Commands (Position Tracking)
        f_cmd_L = arx5.JointState(self.dof)
        f_cmd_L.pos()[:] = l_state_L.pos()
        f_cmd_L.gripper_pos = l_state_L.gripper_pos
        
        
        f_cmd_R = arx5.JointState(self.dof)
        f_cmd_R.pos()[:] = l_state_R.pos()
        f_cmd_R.gripper_pos = l_state_R.gripper_pos


        f_L.set_joint_cmd(f_cmd_L)
        f_R.set_joint_cmd(f_cmd_R)

        # Record Data
        data_packet = {
            'timestamp': current_timestamp,
            'left_q': l_state_L.pos().copy(), 
            'left_gripper': l_state_L.gripper_pos,
        }

        if self.is_recording: self.recorded_data.append(data_packet)

        # Send latest state to Conductor for UI update (throttled)
        if self.state_queue.empty(): self.state_queue.put(pickle.dumps(data_packet))

    def _save_data(self, episode_idx):
        data_dir = "collected_data/robot_logs"
        os.makedirs(data_dir, exist_ok=True)
        filename = os.path.join(data_dir, f"robot_ep_{episode_idx}.pkl")
        with open(filename, 'wb') as f:
            pickle.dump(self.recorded_data, f)