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

# Global variables for graceful shutdown
controller = None
controller2 = None

def stitch_cameras(cameras):
    """Stitch multiple camera images into a single image"""
    if not cameras or len(cameras) == 0:
        return None
    
    # Convert to numpy arrays if needed and ensure consistent shape
    cam_arrays = []
    for cam in cameras:
        if cam is not None:
            cam_arrays.append(cam)
    
    if not cam_arrays:
        return None
    
    # Arrange cameras in a grid (e.g., 2x3 for 5 cameras, with one empty slot)
    num_cams = len(cam_arrays)
    if num_cams <= 2:
        # Horizontal arrangement
        stitched = np.hstack(cam_arrays)
    elif num_cams <= 4:
        # 2x2 grid
        top_row = np.hstack(cam_arrays[:2])
        if num_cams == 3:
            # Pad with zeros for missing camera
            bottom_row = np.hstack([cam_arrays[2], np.zeros_like(cam_arrays[2])])
        else:
            bottom_row = np.hstack(cam_arrays[2:4])
        stitched = np.vstack([top_row, bottom_row])
    else:
        # 2x3 grid for 5+ cameras
        top_row = np.hstack(cam_arrays[:3])
        if num_cams == 5:
            # Pad with zeros for missing camera
            bottom_row = np.hstack([cam_arrays[3], cam_arrays[4], np.zeros_like(cam_arrays[4])])
        else:
            bottom_row = np.hstack(cam_arrays[3:6])
        stitched = np.vstack([top_row, bottom_row])
    
    return stitched

def run_episode_from_zarr(zarr_path, episode_id, robot_model="X5", interface_name="can2", fps=30):
    """
    Reads a zarr file, extracts joint positions for a specific episode, and executes them on the robot.
    Includes graceful shutdown on Control+C and stitches camera outputs into a single image.

    Args:
        zarr_path (str): Path to the zarr file.
        episode_id (int): The episode index to execute.
        robot_model (str): Robot model name.
        interface_name (str): CAN interface name.
    """
    global controller, controller2
    
    # Load replay buffer
    buffer = ReplayBuffer.create_from_path(zarr_path)
    episode = buffer.get_episode(episode_id)

    # joint_pos_arr_R = episode["joint_pos_R"]  # shape: (T, DOF)
    # gripper_pos_arr_R = episode["gripper_pos_R"]  # shape: (T,)
    # joint_pos_arr_L = episode["joint_pos_L"]  # shape: (T, DOF)
    # gripper_pos_arr_L = episode["gripper_pos_L"]  # shape: (T,)
    # camera_obs = episode["color"]  # shape: (T, H, W, C)
    poses = episode["action"]
    cameras = [episode[f"camera{idx}_rgb"] for idx in range(5)]
    print(f"Loaded episode {episode_id} with {poses.shape[0]} steps.")
    # gripper_pos_arr_L = poses[:,6]
    # gripper_pos_arr_R = poses[:,13]
    # print(gripper_pos_arr_L.max(), gripper_pos_arr_L.min())
    # print(gripper_pos_arr_R.max(), gripper_pos_arr_R.min())
    # for cam in episode['camera4_rgb']:
    #     cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
    #     cv2.imshow('RealSense', cam)
    #     cv2.waitKey(1)
    #     time.sleep(1/fps)


    # Initialize controller
    controller = arx5.Arx5JointController(robot_model, interface_name)
    controller2 = arx5.Arx5JointController("X5", "can3")
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
    controller.reset_to_home()
    controller2.reset_to_home()
    
    try:
        for pose, *allcams in zip(poses, *cameras):
            t = time.monotonic()
            cmd = arx5.JointState(6)
            cmd.pos()[:] = pose[0:6]
            cmd.gripper_pos = pose[6]
            controller.set_joint_cmd(cmd)
            cmd2 = arx5.JointState(6)
            cmd2.pos()[:] = pose[7:13]
            cmd2.gripper_pos = pose[13]
            controller2.set_joint_cmd(cmd2)
            elapsed = time.monotonic() - t
            
            # Stitch camera outputs into a single image
            stitched_image = stitch_cameras(allcams)
            if stitched_image is not None:
                cv2.namedWindow('Stitched_Cameras', cv2.WINDOW_AUTOSIZE)
                cv2.imshow('Stitched_Cameras', stitched_image)
                cv2.waitKey(1)
            
            sleep_time = max(0, (1.0 / fps) - elapsed)
            time.sleep(sleep_time)
    except KeyboardInterrupt:
        pass
    finally:
        # Ensure robots are reset to home position
        controller.reset_to_home()
        controller2.reset_to_home()
        cv2.destroyAllWindows()

# Example usage:
if __name__ == "__main__":
    run_episode_from_zarr("/home/ajay/Documents/arx5-sdk/python/data_real/replay_buffer.zarr", episode_id=1)