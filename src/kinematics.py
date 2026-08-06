"""!
Implements Forward and Inverse kinematics with DH parametrs and product of exponentials

TODO: Here is where you will write all of your kinematics functions
There are some functions to start with, you may need to implement a few more
"""

import numpy as np
# expm is a matrix exponential function
from scipy.linalg import expm
from constant import *

def clamp(angle):
    """!
    @brief      Clamp angles between (-pi, pi]

    @param      angle  The angle

    @return     Clamped angle
    """
    while angle > np.pi:
        angle -= 2 * np.pi
    while angle <= -np.pi:
        angle += 2 * np.pi
    return angle

def clamp_half_pi(angle):
    """!
    @brief      Clamp angles between (-pi, pi]

    @param      angle  The angle

    @return     Clamped angle
    """
    while angle > np.pi / 4:
        angle -= np.pi / 2
    while angle <= -np.pi / 4:
        angle += np.pi / 2
    return angle

def FK_dh(dh_params, joint_angles, link):
    """!
    @brief      Get the 4x4 transformation matrix from link to world

                TODO: implement this function

                Calculate forward kinematics for rexarm using DH convention

                return a transformation matrix representing the pose of the desired link

                note: phi is the euler angle about the y-axis in the base frame

    @param      dh_params     The dh parameters as a 2D list each row represents a link and has the format [a, alpha, d,
                              theta]
    @param      joint_angles  The joint angles of the links
    @param      link          The link to transform from

    @return     a transformation matrix representing the pose of the desired link
    """
    pass
    H = np.eye(4)
    for i in range(link):
        T = get_transform_from_dh(dh_params[i][0], dh_params[i][1], dh_params[i][2], clamp(joint_angles[i] + dh_params[i][3]))
        H = np.dot(H, T)
    return H


def get_transform_from_dh(a, alpha, d, theta):
    """!
    @brief      Gets the transformation matrix T from dh parameters.

    TODO: Find the T matrix from a row of a DH table

    @param      a      a meters
    @param      alpha  alpha radians
    @param      d      d meters
    @param      theta  theta radians

    @return     The 4x4 transformation matrix.
    """
    T = np.array([[np.cos(theta), -np.sin(theta) * np.cos(alpha), np.sin(theta) * np.sin(alpha), a * np.cos(theta)],
                    [np.sin(theta), np.cos(theta) * np.cos(alpha), -np.cos(theta) * np.sin(alpha), a * np.sin(theta)],
                    [0, np.sin(alpha), np.cos(alpha), d],
                    [0, 0, 0, 1]])
    return T

def get_euler_angles_from_T(T):
    """!
    @brief      Gets the euler angles from a transformation matrix.

                TODO: Implement this function return the 3 Euler angles from a 4x4 transformation matrix T
                If you like, add an argument to specify the Euler angles used (xyx, zyz, etc.)

    @param      T     transformation matrix

    @return     The euler angles from T.
    """
    from scipy.spatial.transform import Rotation as Rscipy
    rot = Rscipy.from_matrix(T[:3, :3])
    angles = rot.as_euler('ZYZ', degrees=False)
    phi, theta, psi = angles
    return phi, theta, psi


def get_pose_from_T(T):
    """!
    @brief      Gets the pose from T.

                TODO: implement this function return the 6DOF pose vector from a 4x4 transformation matrix T

    @param      T     transformation matrix

    @return     The pose vector from T.
    """
    x = T[0, 3]
    y = T[1, 3]
    z = T[2, 3]
    phi, theta, psi = get_euler_angles_from_T(T)
    phi, theta, psi = clamp(phi), clamp(theta), clamp(psi)
    return [x, y, z, phi, theta, psi]


def FK_pox(joint_angles, m_mat, s_lst):
    """!
    @brief      Get a  representing the pose of the desired link

                TODO: implement this function, Calculate forward kinematics for rexarm using product of exponential
                formulation return a 4x4 homogeneous matrix representing the pose of the desired link

    @param      joint_angles  The joint angles
                m_mat         The M matrix
                s_lst         List of screw vectors

    @return     a 4x4 homogeneous matrix representing the pose of the desired link
    """
    pass


def to_s_matrix(w, v):
    """!
    @brief      Convert to s matrix.

    TODO: implement this function
    Find the [s] matrix for the POX method e^([s]*theta)

    @param      w     { parameter_description }
    @param      v     { parameter_description }

    @return     { description_of_the_return_value }
    """
    pass


def IK_geometric(pose):
    """!
    @brief      Get all possible joint configs that produce the pose.

                TODO: Convert a desired end-effector pose vector as np.array to joint angles

    @param      dh_params  The dh parameters
    @param      pose       The desired pose vector as np.array 

    @return     All four possible joint configurations in a numpy array 4x4 where each row is one possible joint
                configuration
    """
    # x, y, z, psi = pose
    x, y, z = pose
    dist = np.sqrt(x ** 2 + y ** 2)
    if dist < DIST_THRESH_FOR_PSI_90:
        psi = VERTICAL
    elif dist < DIST_THRESH_FOR_PSI_75:
        psi = 5 * np.pi / 12
    elif dist < DIST_THRESH_FOR_PSI_60:
        psi = 1 * np.pi / 3
    elif dist < DIST_THRESH_FOR_PSI_45:
        psi = 1 * np.pi / 4
    else:
        psi = 2 * np.pi / 3
    # print(f"{psi=}")
    # print('psi:', int(psi * 180 / np.pi))
    z += np.cos(psi) * 10 # adjustment for the end effector's height due to different psi
    psi = np.pi * 84 / 180
    while True:
        if psi < 1 * np.pi / 4:
            print("after iteration")
            return None, None, None, None
        print(f'tring psi = {int(psi * 180 / np.pi)}')
        xc  = x - 174.15 * np.cos(psi) * x / np.sqrt(x ** 2 + y ** 2)
        yc  = y - 174.15 * np.cos(psi) * y / np.sqrt(x ** 2 + y ** 2)
        zc  = z + 174.15 * np.sin(psi) 

        y_p = np.sqrt(xc ** 2 + yc ** 2)
        z_p = zc - 103.91

        val = (y_p ** 2 + z_p ** 2 - 82500) / (20000 * np.sqrt(17))
        if val > 1 or val < -1:
            # if the value is out of range, adjust psi a little bit and try again
            psi -= 1 * np.pi / 180
            continue
        q1 = np.arctan2(-xc, yc)
        q2 = np.arctan2(y_p, z_p) - np.arccos((2500 + y_p ** 2 + (z_p) ** 2) / (100 * np.sqrt(17 * (y_p ** 2 + z_p ** 2)))) - np.arctan2(1, 4)
        q3 = np.arccos((y_p ** 2 + z_p ** 2 - 82500) / (20000 * np.sqrt(17))) - np.arctan2(4, 1)

        q4 = psi - q2 - q3

        pass_joint_angle_check = True
        if q1 < np.radians(-180) or q1 > np.radians(180):
            pass_joint_angle_check = False
        if q2 < np.radians(-108) or q2 > np.radians(113):
            pass_joint_angle_check = False
        if q3 < np.radians(-108) or q3 > np.radians(93):
            pass_joint_angle_check = False
        if q4 < np.radians(-180) or q4 > np.radians(123):
            pass_joint_angle_check = False
        
        if pass_joint_angle_check:
            break
        else:
            psi -= 1 * np.pi / 180
    
    # base_offset = - 2 / 180 * np.pi
    base_offset = - 0.2 / 180 * np.pi
    return clamp(q1 + base_offset), clamp(q2), clamp(q3), clamp(q4)