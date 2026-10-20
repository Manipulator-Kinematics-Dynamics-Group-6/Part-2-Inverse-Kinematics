#!/usr/bin/env python3
#coding=utf-8
import time
from Arm_Lib import Arm_Device
Arm = Arm_Device()
time.sleep(.1)
import numpy as np

def invkin_subproblems_Dofbot(Rot, Pot):
    """
    Computes joint angles for Yahboom's Dofbot manipulator using
    canonical subproblems.

    Parameters
    ----------
    Rot : array_like, shape (3, 3)
        End-effector rotation matrix relative to the base frame.
    Pot : array_like, shape (3,) or (3, 1)
        End-effector position relative to the base frame, in meters.

    Returns
    -------
    q : ndarray, shape (5, n)
        Valid joint configurations in degrees, where n is the
        number of solutions found.
    """

    Rot = np.asarray(Rot, dtype=float).reshape(3, 3)
    Pot = np.asarray(Pot, dtype=float).reshape(3)

    # Set up the basis unit vectors
    ex = np.array([1.0, 0.0, 0.0])
    ey = np.array([0.0, 1.0, 0.0])
    ez = np.array([0.0, 0.0, 1.0])

    # Define the link lengths in meters
    l0 = 0.061
    l1 = 0.0435
    l2 = 0.08285
    l3 = 0.08285
    l4 = 0.07385
    l5 = 0.05457

    L1 = l0 + l1
    L4 = l4 + l5

    # Position vectors between subsequent frames
    P01 = (l0 + l1) * ez
    P12 = np.zeros(3)
    P23 = l2 * ex
    P34 = -l3 * ez
    P45 = np.zeros(3)
    P5T = -(l4 + l5) * ex

    # Placeholder for possible joint configurations
    q = np.full((5, 4), np.nan)

    # --------------------------------------------------
    # Get theta = q2 + q3 + q4 from Rot using SP4
    # --------------------------------------------------
    k = -ey
    h = ez
    p = ex
    d = ez.T @ Rot @ ex

    thetatmp = np.atleast_1d(subprob4(k, h, p, d))

    if len(thetatmp) == 1:
        theta = np.array([
            thetatmp[0], np.nan, thetatmp[0], np.nan
        ])
    else:
        theta = np.array([
            thetatmp[0], thetatmp[1],
            thetatmp[0], thetatmp[1]
        ])

    # --------------------------------------------------
    # Get q1 from Rot using SP1
    # --------------------------------------------------
    for ii in range(4):
        if not np.isnan(theta[ii]):
            k = ez
            p1 = rot(ey, -theta[ii]) @ ex
            p2 = Rot @ ex

            q[0, ii] = subprob1(k, p1, p2)

    # --------------------------------------------------
    # Get q5 from Rot using SP1
    # --------------------------------------------------
    for ii in range(4):
        if not np.isnan(theta[ii]):
            k = ex
            p1 = rot(ey, theta[ii]) @ ez
            p2 = Rot.T @ ez

            q[4, ii] = subprob1(k, p1, p2)

    # --------------------------------------------------
    # Get q3 from Pot using SP3
    # --------------------------------------------------
    for ii in range(2):
        if not np.isnan(theta[ii]):
            Pprime = (
                rot(ez, -q[0, ii]) @ (Pot - L1 * ez)
                + rot(ey, -theta[ii]) @ (L4 * ex)
            )

            k = -ey
            p1 = l3 * ez
            p2 = l2 * ex
            d = np.linalg.norm(Pprime)

            q3tmp = np.atleast_1d(subprob3(k, p1, p2, d))

            if len(q3tmp) == 1:
                q[2, ii] = q3tmp[0]
            else:
                q[2, ii] = q3tmp[0]
                q[2, ii + 2] = q3tmp[1]

    # --------------------------------------------------
    # Get q2 from Pot using SP1
    # --------------------------------------------------
    for ii in range(4):
        if not np.isnan(q[2, ii]):
            Pprime = (
                rot(ez, -q[0, ii]) @ (Pot - L1 * ez)
                + rot(ey, -theta[ii]) @ (L4 * ex)
            )

            k = -ey
            p1 = (
                l2 * ex
                - rot(ey, -q[2, ii]) @ (l3 * ez)
            )
            p2 = Pprime

            q[1, ii] = subprob1(k, p1, p2)

    # --------------------------------------------------
    # Get q4 from theta
    # --------------------------------------------------
    for ii in range(4):
        if not np.isnan(q[2, ii]):
            q[3, ii] = theta[ii] - q[1, ii] - q[2, ii]

    # --------------------------------------------------
    # Remove invalid configurations containing NaNs
    # --------------------------------------------------
    valid_columns = ~np.isnan(q).any(axis=0)
    q = q[:, valid_columns]

    # Convert joint angles from radians to degrees
    q = np.rad2deg(q)

    # Normalize angles as in the original MATLAB code
    for ii in range(q.shape[0]):
        for jj in range(q.shape[1]):
            if q[ii, jj] < -180:
                q[ii, jj] += 360
            elif q[ii, jj] > 360:
                q[ii, jj] -= 360

    return q

def main():
    
    # Initialize joint angles and use forward kinematics to calculate the end-effector position and orientation
    Rot = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    Pot = np.array([0.1, 0.1, 0.1])
    inv_kinematics = invkin_subproblems_Dofbot(q)
    
    # Check if the end-effector position is valid
    if (Pot <= 0).any():
        print("Error: The end-effector position is not valid. Please check the joint angles.")
    
    else:
    # Print the calculated angles of all servos
        print("Calculated angles of all servos:")
        for i in range(5):
            print("Servo", i + 1, ":", inv_kinematics[i])
            time.sleep(.01)

    # Command the robot arm to move to the calculated position
    Arm.Arm_serial_servo_write6(inv_kinematics[0], inv_kinematics[1], inv_kinematics[2], inv_kinematics[3], inv_kinematics[4], 180, 2500)

    # Print the actual angles of all servos after the movement
    print("Actual angles of all servos after movement:")
    for i in range(1,7):
        print("Servo", i, ":", Arm.Arm_serial_servo_read(i))
        time.sleep(.01)

try :
    main()
except KeyboardInterrupt:
    print(" Program closed! ")
    pass

del Arm # Release Arm object