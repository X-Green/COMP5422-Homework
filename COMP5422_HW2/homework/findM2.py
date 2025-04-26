'''
Q2.3.3:
    1. Load point correspondences calculated and saved in Q2.2.1
    2. Obtain the correct M2
    3. Save the correct M2, C2, and P to q2.3_3.npz
'''

import numpy as np
import cv2
import matplotlib.pyplot as plt
import submission as sub
import helper


def main():
    data = np.load('q2.2_1.npz')
    points1, points2 = data['points1'], data['points2']

    intrinsics = np.loadtxt('../data/Intrinsic4Recon.npz')
    K1 = intrinsics[0].reshape(3, 3)
    K2 = intrinsics[1].reshape(3, 3)

    F_data = np.load('q2.2_2.npz')
    F = F_data['F']

    E = sub.essentialMatrix(F, K1, K2)

    # M1 = [I|0]
    M1 = np.hstack((np.eye(3), np.zeros((3, 1))))
    M2s = helper.camera2(E)

    C1 = K1 @ M1

    best_count = 0
    best_M2_idx = -1
    best_P = None
    best_C2 = None
    best_error = float('inf')

    # Find BEST
    for i in range(4):
        M2 = M2s[:, :, i]
        C2 = K2 @ M2

        P, err = sub.triangulate(C1, points1, C2, points2)

        # P_homogeneous = np.hstack((P, np.ones((P.shape[0], 1))))

        in_front_of_camera1 = P[:, 2] > 0

        # 对于第二个相机，我们需要将点转换到相机坐标系
        # M2 = [R|t]，所以我们需要计算 R * P + t
        R = M2[:, :3]
        t = M2[:, 3]

        P_in_camera2 = np.zeros_like(P)
        for j in range(P.shape[0]):
            P_in_camera2[j] = R @ P[j] + t

        in_front_of_camera2 = P_in_camera2[:, 2] > 0
        in_front_of_both = np.logical_and(in_front_of_camera1, in_front_of_camera2)
        count = np.sum(in_front_of_both)
        print(f"M2[{i}]: {count}/{len(P)} points in front of CAM，err: {err:.6f}")

        if count > best_count or (count == best_count and err < best_error):
            best_count = count
            best_M2_idx = i
            best_P = P
            best_C2 = C2
            best_error = err

    best_M2 = M2s[:, :, best_M2_idx]
    print(f"\nBest M2 at M2[{best_M2_idx}], with {best_count} p_in_front，err: {best_error:.6f}")

    np.savez('q2.3_3.npz', M2=best_M2, C2=best_C2, P=best_P)
    print("Saved to  q2.3_3.npz")


if __name__ == "__main__":
    main()
