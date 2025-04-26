'''
It is a main function that can be used for you to conduct the tasks in 3D reconstruction.
You run this main function to generate the expected outputs and results described in the Instruction.pdf, 
by calling functions implemented in submission.py and helper.py
You are free to write it in your own style.
'''

# Insert your package here
import numpy as np
import cv2
import matplotlib.pyplot as plt
import submission as sub
import helper

if __name__ == "__main__":
    '''
    Q2.1: Image Matching
    '''
    # Load the two temple images
    img1 = cv2.imread('../data/img1.jpg')
    img2 = cv2.imread('../data/img2.jpg')

    if img1 is None or img2 is None:
        print("Error: Could not read one or both of the input images.")
        exit()

    # Perform feature detection and matching
    points1, points2, matches = sub.matchPoints(img1, img2)

    # Display number of matched points
    print(f"Number of matched points: {len(points1)}")

    # Save matched 2D points to q2.2_1.npz
    np.savez('q2.2_1.npz', points1=points1, points2=points2, matches=matches)
    print("Matched points saved to q2.2_1.npz")

    '''
    Q2.2: Eight-Point Algorithm
    '''
    # Get image dimensions for normalization
    M = max(img1.shape[0], img1.shape[1], img2.shape[0], img2.shape[1])
    print(f"Image Max Size: M: {M}")

    # Compute Fundamental Matrix
    F = sub.eightpoint(points1, points2, M)
    print("Fundamental Matrix F:")
    print(F)
    np.savez('q2.2_2.npz', F=F, M=M)

    # helper.displayEpipolarF(img1, img2, F)

    '''
    Q2.3: Essential Matrix
    '''

    intrinsics = np.loadtxt('../data/Intrinsic4Recon.npz')
    K1 = intrinsics[0].reshape(3, 3)
    K2 = intrinsics[1].reshape(3, 3)

    print("K1 and K2 instrinsics matrices:")
    print(K1)

    E = sub.essentialMatrix(F, K1, K2)
    print("Essential Matrix E:")
    print(E)
    # Using fixed M1 = [I|0]
    M1 = np.hstack((np.eye(3), np.zeros((3, 1))))
    print("Camera Matrix M1 (fixed):")
    print(M1)

    M2s = helper.camera2(E)  # four possible M2 matrices
    print("Four possible Camera Matrices M2:")
    for i in range(4):
        print(f"M2 Option {i + 1}:")
        print(M2s[:, :, i])

    import findM2

    findM2.main()

    '''
    Q2.4: Epipolar Correspondence GUI
    '''
    np.savez('q2.4_1.npz', F=F, pts1=points1, pts2=points2)
    print("Saved F, pts1, pts2 to q2.4_1.npz")

    helper.epipolarMatchGUI(img1, img2, F)
