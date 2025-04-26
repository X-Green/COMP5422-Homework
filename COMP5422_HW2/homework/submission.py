"""
Homework2.
Replace 'pass' by your implementation.
"""

# Insert your package here
import numpy as np
import cv2
import matplotlib.pyplot as plt
from skimage.measure import ransac
from skimage.transform import FundamentalMatrixTransform, AffineTransform
from helper import *
import helper
import os
import time

'''
Q2.1: Image Matching
    Input:  im1, the first image
            im2, the second image
            
    Output: points1, Nx2 matrix of points in image 1
            points2, Nx2 matrix of points in image 2
            matches, Nx1 matrix of indices, where matches[i] indicates the match 
            in the second image for the i-th keypoint in the first image
'''


def matchPoints(im1, im2, visualize=True):
    # Convert images to grayscale if they are not already
    if len(im1.shape) == 3:
        im1_gray = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
    else:
        im1_gray = im1

    if len(im2.shape) == 3:
        im2_gray = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)
    else:
        im2_gray = im2

    # Initialize SIFT detector
    sift = cv2.SIFT_create()

    # Detect keypoints and compute descriptors
    kp1, des1 = sift.detectAndCompute(im1_gray, None)
    kp2, des2 = sift.detectAndCompute(im2_gray, None)

    # Use FLANN matcher for faster matching
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)

    matches = flann.knnMatch(des1, des2, k=2)

    # Apply Lowe's ratio test to filter good matches
    good_matches = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good_matches.append(m)

    # Sort matches by distance
    good_matches = sorted(good_matches, key=lambda x: x.distance)

    # Extract coordinates of matched points
    points1 = np.float32([kp1[m.queryIdx].pt for m in good_matches])
    points2 = np.float32([kp2[m.trainIdx].pt for m in good_matches])

    # Apply RANSAC to filter outliers (optional)
    if len(points1) > 8:  # Need at least 8 points for fundamental matrix estimation
        # Use scikit-image's RANSAC implementation
        model, inliers = ransac(
            (points1, points2),
            FundamentalMatrixTransform,
            min_samples=8,
            residual_threshold=2,
            max_trials=10000
        )

        # Filter points to include only inliers
        points1 = points1[inliers]
        points2 = points2[inliers]
        # Keep track of which matches were inliers
        good_matches = [good_matches[i] for i in range(len(good_matches)) if inliers[i]]

    if visualize:
        # Visualize the keypoints
        img_kp1 = cv2.drawKeypoints(im1, kp1, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        img_kp2 = cv2.drawKeypoints(im2, kp2, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.imshow(cv2.cvtColor(img_kp1, cv2.COLOR_BGR2RGB))
        plt.title('Keypoints in Image 1')
        plt.subplot(1, 2, 2)
        plt.imshow(cv2.cvtColor(img_kp2, cv2.COLOR_BGR2RGB))
        plt.title('Keypoints in Image 2')
        plt.savefig('keypoints.png')

        # Draw top 100 matches
        top_matches = good_matches[:100] if len(good_matches) > 100 else good_matches
        matches_img = cv2.drawMatches(im1, kp1, im2, kp2, top_matches, None,
                                      flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

        plt.figure(figsize=(12, 6))
        plt.imshow(cv2.cvtColor(matches_img, cv2.COLOR_BGR2RGB))
        plt.title('Top Matches Between Images')
        plt.savefig('matches.png')

    # Create match indices
    matches_idx = np.arange(len(points1))

    return points1, points2, matches_idx


'''
Q2.2.2: Eight Point Algorithm
    Input:  pts1, Nx2 Matrix
            pts2, Nx2 Matrix
            M, a scalar parameter computed as max (imwidth, imheight)
    Output: F, the fundamental matrix
'''


def eightpoint(pts1, pts2, M):
    # 1. 归一化点坐标
    normalized_pts1 = pts1 / M
    normalized_pts2 = pts2 / M

    T = np.array([
        [1 / M, 0, 0],
        [0, 1 / M, 0],
        [0, 0, 1]
    ])

    n = pts1.shape[0]

    # 2. 建立约束方程组 Af = 0
    A = np.zeros((n, 9))
    for i in range(n):
        x1, y1 = normalized_pts1[i]
        x2, y2 = normalized_pts2[i]
        A[i] = [x1 * x2, y1 * x2, x2, x1 * y2, y1 * y2, y2, x1, y1, 1]

    _, _, Vt = np.linalg.svd(A)
    F = Vt[-1]
    F = F.reshape(3, 3)

    #  强制 F 为秩 2 的矩阵（奇异性约束）
    U, S, Vt = np.linalg.svd(F)
    S[2] = 0
    # S'= [s1, s2, 0]
    F = U @ np.diag(S) @ Vt
    # F' = US'Vt

    F = refineF(F, normalized_pts1, normalized_pts2)

    # 反归一化 F: F_unnormalized = T^T * F * T
    F_denormalized = T.T @ F @ T

    return F_denormalized


'''
Q2.3.1: Compute the essential matrix E.
    Input:  F, fundamental matrix
            K1, internal camera calibration matrix of camera 1
            K2, internal camera calibration matrix of camera 2
    Output: E, the essential matrix
'''


def essentialMatrix(F, K1, K2):
    # Given intrinsic matrices K1 and K2
    E = K2.T @ F @ K1

    # Ensure E is rank 2
    U, S, Vt = np.linalg.svd(E)
    sigma = (S[0] + S[1]) / 2
    S = np.array([sigma, sigma, 0])
    E = U @ np.diag(S) @ Vt

    return E


'''
Q2.3.2: Triangulate a set of 2D coordinates in the image to a set of 3D points.
    Input:  C1, the 3x4 camera matrix
            pts1, the Nx2 matrix with the 2D image coordinates per row
            C2, the 3x4 camera matrix
            pts2, the Nx2 matrix with the 2D image coordinates per row
    Output: P, the Nx3 matrix with the corresponding 3D points per row
            err, the reprojection error.
'''


def triangulate(C1, pts1, C2, pts2):
    n = pts1.shape[0]  # 点的数量
    P = np.zeros((n, 3))  # 结果：n个3D点

    for i in range(n):
        u1, v1 = pts1[i]
        u2, v2 = pts2[i]

        # 构建矩阵A（大小为4x4）
        A = np.zeros((4, 4))
        A[0] = u1 * C1[2] - C1[0]  # u1 * (C1第3行) - (C1第1行)
        A[1] = v1 * C1[2] - C1[1]  # v1 * (C1第3行) - (C1第2行)
        A[2] = u2 * C2[2] - C2[0]  # u2 * (C2第3行) - (C2第1行)
        A[3] = v2 * C2[2] - C2[1]  # v2 * (C2第3行) - (C2第2行)

        # 使用SVD求解 A.P = 0
        _, _, Vt = np.linalg.svd(A)
        P_i_homogeneous = Vt[-1]  # 取V的最后一列作为齐次最小二乘解

        # 转换为非齐次坐标 (x,y,z)
        P[i] = P_i_homogeneous[:3] / P_i_homogeneous[3]

    # 计算重投影误差
    total_error = 0

    for i in range(n):
        # 将3D点转换为齐次坐标
        P_i_homogeneous = np.append(P[i], 1)

        # 投影到两个图像
        p1_proj_homogeneous = C1 @ P_i_homogeneous
        p2_proj_homogeneous = C2 @ P_i_homogeneous

        # to 2D Cartesian coordinates
        p1_proj = p1_proj_homogeneous[:2] / p1_proj_homogeneous[2]
        p2_proj = p2_proj_homogeneous[:2] / p2_proj_homogeneous[2]

        # error1 = np.linalg.norm(p1_proj - pts1[i])
        # error2 = np.linalg.norm(p2_proj - pts2[i])
        error1 = np.sum((p1_proj - pts1[i]) ** 2)
        error2 = np.sum((p2_proj - pts2[i]) ** 2)

        total_error += (error1 + error2)

    err = total_error / (2 * n)

    return P, err


'''
Q2.4.1: 3D visualization of the temple images.
    Input:  im1, the first image
            im2, the second image
            F, the fundamental matrix
            x1, x-coordinates of a pixel on im1
            y1, y-coordinates of a pixel on im1
    Output: x2, x-coordinates of the pixel on im2
            y2, y-coordinates of the pixel on im2

'''


def epipolarCorrespondence(im1, im2, F, x1, y1):
    # 将坐标转为整数
    x1, y1 = int(round(x1)), int(round(y1))

    # 转换为灰度图并转为浮点型
    if len(im1.shape) == 3:
        im1_gray = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
    else:
        im1_gray = im1
    if len(im2.shape) == 3:
        im2_gray = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)
    else:
        im2_gray = im2

    im1_gray = im1_gray.astype(float)
    im2_gray = im2_gray.astype(float)

    # 定义窗口大小和搜索范围
    window_size = 11
    search_radius = 30  # 搜索范围
    max_line_dist = 1.0  # 到极线的最大距离阈值

    half_window = window_size // 2

    # 创建高斯核进行加权
    gaussian_kernel = cv2.getGaussianKernel(window_size, sigma=window_size / 6)
    gaussian_kernel_2d = gaussian_kernel @ gaussian_kernel.T

    # 填充图像以处理边界
    im1_padded = cv2.copyMakeBorder(im1_gray, half_window, half_window, half_window, half_window, cv2.BORDER_REFLECT)
    im2_padded = cv2.copyMakeBorder(im2_gray, half_window, half_window, half_window, half_window, cv2.BORDER_REFLECT)

    # 获取参考窗口 - 注意坐标调整
    x1_pad = x1 + half_window
    y1_pad = y1 + half_window
    ref_window = im1_padded[y1_pad - half_window:y1_pad + half_window + 1,
                 x1_pad - half_window:x1_pad + half_window + 1]
    weighted_ref_window = ref_window * gaussian_kernel_2d

    # 计算im2中的极线
    p1_homogeneous = np.array([[x1], [y1], [1]])
    epipolar_line = F @ p1_homogeneous  # l = [a, b, c] 满足 ax + by + c = 0
    a, b, c = epipolar_line.flatten()
    line_norm = np.sqrt(a ** 2 + b ** 2)

    min_error = float('inf')
    best_x2, best_y2 = -1, -1

    # 定义im2中的搜索范围
    y_min = max(half_window, y1 - search_radius)
    y_max = min(im2_gray.shape[0] - 1 - half_window, y1 + search_radius)

    # 搜索im2中沿极线的最佳匹配点
    for y2 in range(y_min, y_max + 1):
        # 计算在当前y2值下，极线上对应的x2值 (ax + by + c = 0 => x = -(by + c)/a)
        if abs(a) > 1e-9:  # 避免除以0
            x2 = int(round(-(b * y2 + c) / a))
            # 检查x2是否在图像范围内
            if x2 < half_window or x2 >= im2_gray.shape[1] - half_window:
                continue

            # 在x2附近的小范围内搜索最佳匹配
            x_range = range(max(half_window, x2 - 5), min(im2_gray.shape[1] - 1 - half_window, x2 + 6))

            for x_c in x_range:
                # 计算点到极线的距离
                dist_to_line = abs(a * x_c + b * y2 + c) / line_norm
                if dist_to_line > max_line_dist:  # 超过阈值则跳过
                    continue

                # 获取候选窗口 - 注意坐标调整
                x_c_pad = x_c + half_window
                y2_pad = y2 + half_window
                cand_window = im2_padded[y2_pad - half_window:y2_pad + half_window + 1,
                              x_c_pad - half_window:x_c_pad + half_window + 1]
                weighted_cand_window = cand_window * gaussian_kernel_2d

                # 计算相似度 (SSD)
                error = np.sum((weighted_ref_window - weighted_cand_window) ** 2)

                # 更新最佳匹配
                if error < min_error:
                    min_error = error
                    best_x2, best_y2 = x_c, y2

    if best_x2 == -1:
        print(f"警告: 在极线附近未找到({x1}, {y1})的匹配点。返回原始坐标。")
        return x1, y1

    return best_x2, best_y2


'''
Q3.1: Decomposition of the essential matrix to rotation and translation.
    Input:  im1, the first image
            im2, the second image
            k1, camera intrinsic matrix of the first frame
            k1, camera intrinsic matrix of the second frame
    Output: R, rotation
            r, translation

'''


def essentialDecomposition(im1, im2, k1, k2):
    pts1, pts2, _ = matchPoints(im1, im2, visualize=False)

    h1, w1 = im1.shape[:2]
    M = np.max([w1, h1])

    F = eightpoint(pts1, pts2, M)

    E = essentialMatrix(F, k1, k2)

    M2s = helper.camera2(E)  # 获取四种可能的相机矩阵

    C1 = np.hstack((np.eye(3), np.zeros((3, 1))))
    C1 = k1 @ C1

    # 7. 验证四个解，找出正确的R和t
    max_positive_depths = -1
    best_count = 0
    best_R = None
    best_t = None
    best_P = None
    best_error = float('inf')

    for i in range(4):
        M2 = M2s[:, :, i]
        C2 = k2 @ M2

        P, err = sub.triangulate(C1, pts1, C2, pts2)

        # P_homogeneous = np.hstack((P, np.ones((P.shape[0], 1))))

        in_front_of_camera1 = P[:, 2] > 0

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
            best_error = err
            best_R = R
            best_t = t
    # 返回最佳解
    return best_R, best_t


'''
Q3.2: Implement a monocular visual odometry.
    Input:  datafolder, the folder of the provided monocular video sequence
            GT_pose, the provided ground-truth (GT) pose for each frame
            plot=True, draw the estimated and the GT camera trajectories in the same plot
    Output: trajectory, the estimated camera trajectory (with scale aligned)        

'''


def visualOdometry(datafolder, GT_Pose, plot=True):
    intrinsics = np.loadtxt('../data/Intrinsic4Recon.npz')
    K = intrinsics[0].reshape(3, 3)

    img_files = sorted(
        [os.path.join(datafolder, f) for f in os.listdir(datafolder) if f.endswith('.jpg') or f.endswith('.png')])
    num_frames = len(img_files)
    # num_frames = 300

    trajectory = [np.zeros(3)]  # 初始位置为原点 [0, 0, 0]
    R_curr = np.eye(3)  # I
    t_curr = np.zeros(3)  # 0

    gt_translations = [GT_Pose[i, :3, 3] for i in range(GT_Pose.shape[0])]

    im2 = cv2.imread(img_files[0])

    for i in range(num_frames - 2):
        print(f"Processing frame {i + 1}/{num_frames}")
        start_time_ms = time.time_ns() / 1_000_000

        im1 = im2
        im2 = cv2.imread(img_files[i + 1])
        print(f"File Loaded{img_files[i + 1]} @",time.time_ns() / 1_000_000 - start_time_ms)
        if im1 is None or im2 is None:
            raise f"Error reading images: {img_files[i]} or {img_files[i + 1]}"

        R_rel, t_rel = essentialDecomposition(im1, im2, K, K)

        print(f"Frame {i + 1} R_rel:\n{R_rel}, \nEular:{np.degrees(cv2.Rodrigues(R_rel)[0].flatten())}")
        print(f"Frame {i + 1} t_rel:\n{t_rel}")
        print(f"EssenDecom@",time.time_ns() / 1_000_000 - start_time_ms)

        scale = getAbsoluteScale(gt_translations[i], gt_translations[i + 1])
        print(f"Scale: {scale}")
        # t_update = scale * (R_rel @ t_rel)
        t_update = scale * (R_curr @ t_rel)  # R_curr 是 R_i^W
        t_update[2] = -t_update[2]
        print(f"My t_update:\n {t_update}")
        print(f"GT t_update:\n {gt_translations[i + 1] - gt_translations[i]}")

        t_curr = t_curr + t_update
        R_curr = R_curr @ R_rel

        trajectory.append(t_curr.copy())
        print(f"==================================@",time.time_ns() / 1_000_000 - start_time_ms)

    trajectory = np.array(trajectory)

    if plot:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        # 绘制估计轨迹
        ax.plot(trajectory[:, 0], trajectory[:, 1], trajectory[:, 2], 'b-', label='Estimated Trajectory')
        ax.scatter(trajectory[0, 0], trajectory[0, 1], trajectory[0, 2], c='b', marker='s', label='Start (Est)')
        ax.scatter(trajectory[-1, 0], trajectory[-1, 1], trajectory[-1, 2], c='b', marker='o', label='End (Est)')

        # 绘制 GT 轨迹
        gt_traj = np.array(gt_translations)
        ax.plot(gt_traj[:, 0], gt_traj[:, 1], gt_traj[:, 2], 'r-', label='Ground Truth Trajectory')
        ax.scatter(gt_traj[0, 0], gt_traj[0, 1], gt_traj[0, 2], c='r', marker='s', label='Start (GT)')
        ax.scatter(gt_traj[-1, 0], gt_traj[-1, 1], gt_traj[-1, 2], c='r', marker='o', label='End (GT)')

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title('Visual Odometry Trajectory')
        ax.legend()
        # 确保轴比例一致
        max_range = np.array([trajectory[:, 0].max() - trajectory[:, 0].min(),
                              trajectory[:, 1].max() - trajectory[:, 1].min(),
                              trajectory[:, 2].max() - trajectory[:, 2].min()]).max() / 2.0

        mid_x = (trajectory[:, 0].max() + trajectory[:, 0].min()) * 0.5
        mid_y = (trajectory[:, 1].max() + trajectory[:, 1].min()) * 0.5
        mid_z = (trajectory[:, 2].max() + trajectory[:, 2].min()) * 0.5
        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)

        plt.savefig('visual_odometry_trajectory.png')
        plt.show()

    return trajectory
