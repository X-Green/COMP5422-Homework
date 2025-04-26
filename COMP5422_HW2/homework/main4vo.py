import numpy as np
import os
import cv2
from submission import visualOdometry

def run_visual_odometry():
    # 修正数据文件夹路径 - 指向包含jpg图像的确切位置
    data_folder = '../data/vid_seq/data'
    gt_pose_file = '../data/GTPoses.npz'
    gt_pose_data_flat = np.loadtxt(gt_pose_file)
    num_poses = gt_pose_data_flat.shape[0]
    gt_poses_reshaped = gt_pose_data_flat.reshape(num_poses, 3, 4)

    print("Running Visual Odometry...")
    estimated_trajectory = visualOdometry(data_folder, gt_poses_reshaped, plot=True)

    output_file = 'q3_2.npz'
    try:
        np.savez(output_file, trajectory=estimated_trajectory)
        print(f"Estimated trajectory saved to {output_file}")
    except Exception as e:
        print(f"Error saving trajectory to {output_file}: {e}")
        raise

if __name__ == "__main__":
    run_visual_odometry()
