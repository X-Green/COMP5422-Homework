'''
Q2.4.2:
    1. Integrating everything together.
    2. Loads necessary files from ../data/ and visualizes 3D reconstruction using scatter
'''
import numpy as np
import matplotlib.pyplot as plt
import submission as sub
import helper
import cv2
from mpl_toolkits.mplot3d import Axes3D


def main():
    img1 = cv2.imread('../data/img1.jpg')
    img2 = cv2.imread('../data/img2.jpg')

    data_F = np.load('q2.2_2.npz')
    F = data_F['F']

    pts1 = np.loadtxt('../data/VisPts.npz')
    if pts1.shape[1] != 2:
        raise ValueError("Loaded data does not have 2 columns")
    print(f"Loaded {pts1.shape[0]} points from VisPts.npz for img1.")

    # Find corresponding points in img2 using epipolar correspondence
    print("Finding corresponding points in img2...")
    pts2 = np.zeros_like(pts1)
    for i in range(pts1.shape[0]):
        x1, y1 = pts1[i, 0], pts1[i, 1]
        x2, y2 = sub.epipolarCorrespondence(img1, img2, F, x1, y1)
        pts2[i, 0], pts2[i, 1] = x2, y2

    # Load intrinsic camera parameters
    intrinsics_data = np.loadtxt('../data/Intrinsic4Recon.npz')
    K1 = intrinsics_data[0].reshape(3, 3)
    K2 = intrinsics_data[1].reshape(3, 3)

    print("Camera Intrinsics K1:")
    print(K1)
    print("Camera Intrinsics K2:")
    print(K2)

    # 6. Compute Essential Matrix E
    E = sub.essentialMatrix(F, K1, K2)
    print("Essential Matrix E computed:")
    print(E)
    
    # 获得可能的摄像机矩阵
    M2s = helper.camera2(E)
    
    # 第一个摄像机矩阵固定为 [I|0]
    C1 = np.hstack((np.eye(3), np.zeros((3, 1))))
    
    # 选择正确的摄像机矩阵
    best_C2 = None
    max_points_in_front = 0
    best_P = None
    best_err = float('inf')
    
    for i in range(M2s.shape[2]):
        C2_candidate = M2s[:, :, i]
        # 转换为完整的摄像机矩阵
        cam1 = K1 @ C1
        cam2 = K2 @ C2_candidate
        
        # 三角测量
        P, err = sub.triangulate(cam1, pts1, cam2, pts2)
        
        # 计算在两个摄像机前方的点数量
        # 对于第一个摄像机，z坐标为正表示在前方
        # 对于第二个摄像机，需要将点变换到摄像机坐标系，第三个坐标为正表示在前方
        
        # 计算在第一个摄像机前方的点
        points_in_front1 = sum(P[:, 2] > 0)
        
        # 计算在第二个摄像机前方的点
        # 转换为齐次坐标
        P_homogeneous = np.hstack((P, np.ones((P.shape[0], 1))))
        # 计算在第二个摄像机坐标���中的坐标
        P_cam2 = (C2_candidate @ P_homogeneous.T).T
        points_in_front2 = sum(P_cam2[:, 2] > 0)
        
        # 计算总共在前方的点数
        points_in_front = points_in_front1 + points_in_front2
        
        print(f"Camera matrix {i+1}: {points_in_front} points in front, error: {err}")
        
        if points_in_front > max_points_in_front or (points_in_front == max_points_in_front and err < best_err):
            max_points_in_front = points_in_front
            best_C2 = C2_candidate
            best_P = P
            best_err = err
    
    C2 = best_C2
    P = best_P
    
    print(f"Selected camera matrix with {max_points_in_front} points in front, error: {best_err}")
    print("Camera Matrix C1:")
    print(C1)
    print("Camera Matrix C2:")
    print(C2)

    # --- Outlier Detection ---
    # Calculate the centroid of the point cloud
    centroid = np.mean(P, axis=0)
    # Calculate the Euclidean distance of each point from the centroid
    distances = np.linalg.norm(P - centroid, axis=1)
    
    # Use Interquartile Range (IQR) to determine outliers
    Q1 = np.percentile(distances, 25)
    Q3 = np.percentile(distances, 75)
    IQR = Q3 - Q1
    # Define outlier threshold (e.g., points beyond 1.5 * IQR from Q3)
    outlier_threshold = Q3 + 1.5 * IQR
    
    # Create a mask for inliers
    inlier_mask = distances <= outlier_threshold
    P_inliers = P[inlier_mask]
    
    print(f"Original number of points: {P.shape[0]}")
    print(f"Number of inlier points after outlier removal: {P_inliers.shape[0]}")
    # --- End Outlier Detection ---

    # 可视化3D点云 (使用内点)
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    # Use P_inliers for scatter plot
    ax.scatter(P_inliers[:, 0], P_inliers[:, 1], P_inliers[:, 2], c='b', marker='o', s=10, label='Inlier Points')
    
    # 绘制摄像机位置和方向
    # 摄像机1位于原点
    ax.scatter(0, 0, 0, c='r', marker='^', s=100, label='Camera 1')
    
    # 摄像机2位置从C2的平移部分获得
    cam2_pos = -np.linalg.inv(C2[:, :3]) @ C2[:, 3] if C2[:, :3].size > 0 else np.zeros(3)
    ax.scatter(cam2_pos[0], cam2_pos[1], cam2_pos[2], c='g', marker='^', s=100, label='Camera 2')
    
    # --- 设置固定的坐标轴范围 ---
    ax.set_xlim(-1.5, 3.5)
    ax.set_ylim(-1.5, 0.5)
    ax.set_zlim(0, 20)
    # --- 结束设置固定的坐标轴范围 ---

    # 设置坐标轴标签和视角
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    # Change elevation and azimuth for a different perspective
    ax.view_init(elev=10, azim=-60) # Example: Lower elevation, different azimuth
    plt.title('3D Reconstruction (Inliers Only) - Fixed Range') # Update title
    plt.legend()
    
    # 保存图像
    plt.savefig('3d_reconstruction_inliers_fixed_range.png') # Save with a new name
    plt.show()
    
    # 保存 F, C1, C2 到 q2.4_2.npz
    # 获取完整的摄像机矩阵
    full_C1 = K1 @ C1
    full_C2 = K2 @ C2
    
    np.savez('q2.4_2.npz', F=F, C1=full_C1, C2=full_C2)
    print("Saved matrices F, C1, C2 to q2.4_2.npz")


if __name__ == '__main__':
    main()

