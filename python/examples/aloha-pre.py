import time

import os
import sys

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)
os.chdir(ROOT_DIR)
import arx5_interface as arx5
import click
import numpy as np


# Kc = 0.18         # Coulomb friction estimate (N·m). Start 0.05-0.3
# Ks = 0.30         # Static (breakaway) friction peak (N·m). >= Kc
# B  = 0.45         # Viscous coefficient (N·m / (rad/s)). Start small
# vs = 0.02         # Stribeck velocity (rad/s). Small value controls transition
# vdead = 0.01  # velocity deadband (rad/s) to avoid jitter
max_torque = 1.5 / 2 - 1e-2  # safety clamp (N·m) — set to a safe value for your gripper
# # --------------------------------

# def stribeck_scaling(v, vs):
#     # smooth transition from static->coulomb (1.0 -> 0.0 with increasing v)
#     # using an exponential like classical Stribeck
#     return np.exp(-(abs(v) / vs)**2)

K_COULOMB = 0.12  # Nm (Assists your push)
K_VISCOUS = 0.07 # Nm/(rad/s)

VEL_DEADBAND = 0.02 # rad/s (Ignore small accidental movements)

def sign(x):
    return 1.0 if x > 0 else -1.0 if x < 0 else 0.0
def friction_compensation(current_vel):
    # vel: joint velocity (rad/s)
    print(current_vel)
    if abs(current_vel) > VEL_DEADBAND:
        return (K_COULOMB * sign(current_vel)) + (K_VISCOUS * current_vel)
    else:
        return 0.0

@click.command()
@click.argument("leader_model")  # ARX arm model: X5 or L5
@click.argument("leader_interface")  # can bus name (can0 etc.)
@click.argument("leader_interface2")  # can bus name (can0 etc.)
@click.argument("follower_model")  # ARX arm model: X5 or L5
@click.argument("follower_interface")  # can bus name (can0 etc.)
@click.argument("follower_interface2")  # can bus name (can0 etc.)
def main(leader_model: str, leader_interface: str, leader_interface2, follower_model, follower_interface, follower_interface2):
    np.set_printoptions(precision=3, suppress=True)
    assert(leader_interface != follower_interface)

    leader = arx5.Arx5JointController(leader_model, leader_interface)
    leader2 = arx5.Arx5JointController(leader_model, leader_interface2)

    follower = arx5.Arx5JointController(follower_model, follower_interface)
    follower2 = arx5.Arx5JointController(follower_model, follower_interface2)
    robot_config = leader2.get_robot_config()
    assert robot_config.joint_dof == follower.get_robot_config().joint_dof
    controller_config = leader2.get_controller_config()

    leader.reset_to_home()
    follower.reset_to_home()
    leader2.reset_to_home()
    follower2.reset_to_home()
    gain = arx5.Gain(robot_config.joint_dof)
    # print("*"*10, leader.get_gain().kd(), "*"*10)
    gain.kd()[:] = 0.01
    # gain.kp()[:] = 1.0
    # leader.set_gain(gain)
    # gain = arx5.Gain(robot_config.joint_dof)
    # gain.kd()[:] = 0.01
    leader.set_gain(gain)
    gain = arx5.Gain(robot_config.joint_dof)
    print("*"*10, leader2.get_gain().kd(), "*"*10)
    gain.kd()[:] = 0.01
    # gain.kp()[:] = 1.0
    gain.gripper_kp = 0.0
    gain.gripper_kd = 0.0
    leader2.set_gain(gain)
    print("*"*10, leader2.get_gain().kp(), "*"*10, controller_config.gravity_compensation)
    try:
        while True:
            leader_joint_state = leader.get_joint_state()
            leader_joint_state2 = leader2.get_joint_state()
            # vel = leader_joint_state2.gripper_vel   # assume rad/s or same units expected
            # # print(vel)
            # # # Compute compensation torque
            # tau = friction_compensation(vel)

            # # # # Optional: if gripper is almost closed, set tau=0 (safety)
            # # # if leader_joint_state2.gripper_pos < 0.01:  # width threshold from your code
            # # #     tau = 0.0

            # # # # Clamp torque to safe limits
            # tau = float(np.clip(tau, -max_torque, max_torque))
            # # # print(tau)
            # # # # publish command
            # joint_cmd = arx5.JointState(robot_config.joint_dof)
            # joint_cmd.gripper_torque = tau
            # leader2.set_joint_cmd(joint_cmd)
            # print(leader_joint_state2.gripper_pos)

            # follower_joint_state = follower.get_joint_state()

            # print(leader_joint_state.pos(), leader_joint_state.gripper_pos)
            # print(follower_joint_state.pos(), follower_joint_state.gripper_pos)

            follower_joint_cmd = arx5.JointState(robot_config.joint_dof)
            follower_joint_cmd.pos()[:] = leader_joint_state.pos()
            follower_joint_cmd.gripper_pos = leader_joint_state.gripper_pos

            follower_joint_cmd2 = arx5.JointState(robot_config.joint_dof)
            follower_joint_cmd2.pos()[:] = leader_joint_state2.pos()
            follower_joint_cmd2.gripper_pos = leader_joint_state2.gripper_pos
            # print(follower_joint_cmd2.gripper_pos)
            # # If you want to decrease the delay of teleoperation , you can uncomment the following line
            # # This will partially include the velocity to the command, and you will need to pass these velocities
            # # into the policy layout
            # # follower_joint_cmd.vel()[:] = leader_joint_state.vel()* 0.3 
            
            follower.set_joint_cmd(follower_joint_cmd)
            follower2.set_joint_cmd(follower_joint_cmd2)

            time.sleep(controller_config.controller_dt)
    except KeyboardInterrupt:
        print("\nKeyboard interrupt detected. Resetting arms to home position...")
        follower.reset_to_home()
        leader.reset_to_home()
        follower2.reset_to_home()
        leader2.reset_to_home()
        print("Arms reset to home position. Exiting.")


    



main()
