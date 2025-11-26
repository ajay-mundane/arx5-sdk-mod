import zarr
import numpy as np
import os 
import sys
import time
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)
os.chdir(ROOT_DIR)
import arx5_interface as arx5
from helper import ReplayBuffer
import cv2

def run_episode_from_zarr(zarr_path, episode_id, robot_model="X5", interface_name="can3", fps=30):
    """
    Reads a zarr file, extracts joint positions for a specific episode, and executes them on the robot.

    Args:
        zarr_path (str): Path to the zarr file.
        episode_id (int): The episode index to execute.
        robot_model (str): Robot model name.
        interface_name (str): CAN interface name.
    """
    # Load replay buffer
    buffer = ReplayBuffer.create_from_path(zarr_path)
    episode = buffer.get_episode(episode_id)

    joint_pos_arr_R = episode["joint_pos_R"]  # shape: (T, DOF)
    gripper_pos_arr_R = episode["gripper_pos_R"]  # shape: (T,)
    joint_pos_arr_L = episode["joint_pos_L"]  # shape: (T, DOF)
    gripper_pos_arr_L = episode["gripper_pos_L"]  # shape: (T,)
    camera_obs = episode["color"]  # shape: (T, H, W, C)

    # Initialize controller
    controller = arx5.Arx5JointController(robot_model, interface_name)
    controller2 = arx5.Arx5JointController("X5", "can2")
    gain = arx5.Gain(6)
    gain.kd()[:] = 0.01
    gain.gripper_kp = 0.0
    gain.gripper_kd = 0.0
    controller.set_gain(gain)

    gain = arx5.Gain(6)
    gain.kd()[:] = 0.01
    gain.gripper_kp = 0.0
    gain.gripper_kd = 0.0
    controller2.set_gain(gain)
    controller_config = controller.get_controller_config()
    dof = joint_pos_arr_R.shape[1]
    controller.reset_to_home()
    controller2.reset_to_home()
    for cam, posR, gripperR, posL, gripperL in zip(camera_obs, joint_pos_arr_R, gripper_pos_arr_R, joint_pos_arr_L, gripper_pos_arr_L):
        cmd = arx5.JointState(dof)
        cmd.pos()[:] = posR
        cmd.gripper_pos = gripperR
        controller.set_joint_cmd(cmd)
        cmd = arx5.JointState(dof)
        cmd.pos()[:] = posL
        cmd.gripper_pos = gripperL
        controller2.set_joint_cmd(cmd)
        # controller.send_recv_once()
        # Add a small delay for safety
        cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
        cv2.imshow('RealSense', cam)
        cv2.waitKey(1)
        time.sleep(1/fps)

    controller.reset_to_home()
    controller2.reset_to_home()
# Example usage:
run_episode_from_zarr("/home/ajay/Documents/arx5-sdk/python/data/replay_buffer.zarr", episode_id=0)