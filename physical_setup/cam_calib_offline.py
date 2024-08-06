
#!/usr/bin/env python
from rigid_trans_fitter3D import RigidTransFitter3D
import numpy as np
import matplotlib.pyplot as plt

# points coordinates in camera frame
# cam3D = [
#     [-0.012, 0.090, 0.922],
#     [0.130, 0.185, 0.934],
#     [0.137, 0.046, 0.939],
#     [-0.024, 0.205, 0.928]
# ]
# # points coordinates in robot frame
# rob3D = [
#     [0.167, -0.459, 0.007],
#     [0.117, -0.347, 0.008],
#     [0.112, -0.484, 0.007],
#     [0.268, -0.324, 0.007]
# ]

from computer_vision import D435

d435 = D435()

color_intrinsics = d435.get_entrinsics()

# take 20 coords
cam_coords = [[316, 203, 537],
              [244, 250, 419],
              [341, 123, 412],
              [415, 193, 595],
              [226, 234, 642],
              [198, 139, 607],
              [308, 200, 630],
              [184, 246, 648],
              [254, 199, 343],
              [205, 161, 554],
              [444, 244, 502],
              [379, 208, 381],
              [469, 216, 495],
              [333, 169, 435],
              [244, 148, 520],
              [316, 155, 613],
              [194, 183, 528],
              [287, 256, 641],
              [252, 238, 740],
              [280, 189, 446]]

robot_coords = [[253.8, -492.15, 379.5],
                [317.11, -453.95, 494.14],
                [232.37, -570.9, 501.67],
                [130.55, -507.4, 324.1],
                [380.22, -444.11, 275.7],
                [413.8, -570.5, 312.15],
                [264.9, -493.35, 290.15],
                [436.18, -425.56, 272.3],
                [295.1, -503.57, 573.0],
                [385.25, -539.63, 366.7],
                [114.25, -456.36, 412.27],
                [197.34, -499.12, 533.05],
                [88.5, -484.5, 421.11],
                [234.55, -532.5, 482.05],
                [333.87, -555.1, 397.35],
                [254.5, -552.28, 308.5],
                [388.55, -514.64, 388.88],
                [293.77, -415.05, 277.73],
                [358.47, -428.8, 185.72],
                [286.3, -511.1, 468.8]]

cam3D = []

for i in cam_coords:
    cam3D.append([((i[0] - color_intrinsics.ppx) * (i[2]/1000)) / color_intrinsics.fx, ((i[1] - color_intrinsics.ppy) * (i[2]/1000)) / color_intrinsics.fy, i[2]/1000])

print(cam3D)

# Define the alignment matrix
align_matrix = np.array([[-1, 0, 0, 0],
                         [0, 1, 0, 0],
                         [0, 0, -1, 0],
                         [0, 0, 0, 1]])

# Transform camera coordinates
cam3D_aligned = []
for point in cam3D:
    # Convert point to homogeneous coordinates
    point_homogeneous = np.append(point, 1)
    # Apply alignment transformation
    aligned_point = np.dot(align_matrix, point_homogeneous)
    # Convert back to 3D coordinates and append to list
    cam3D_aligned.append(aligned_point[:3].tolist())

cam3D = cam3D_aligned
#print(cam3D)

rob3D = []

for i in robot_coords:
    rob3D.append([i[0]/1000, i[1]/1000, i[2]/1000])

print(rob3D)

# calibration using RigidTransFitter3D
calibration = RigidTransFitter3D()
tran_cam2rob = calibration.get_transform(cam3D, rob3D)  # check order for feedbing points
print('\ncalculating the transformation matrix:')
print(tran_cam2rob)

# pull out rotation matrix and translation vector
#rot_cam2rob = tran_cam2rob[0:3, 0:3]
#tran_cam2rob = tran_cam2rob[0:3, 3]
#print(tran_cam2rob)
#rot_cam3rob_trans = np.transpose(rot_cam2rob)

#cam_pos_world = -rot_cam3rob_trans * np.transpose(tran_cam2rob)

#print("pos in world frame: ")
#print(cam_pos_world)

# check the calibration results by:
# step 1. calculating the cam3D points in robot frame using calibration matrix
# step 2. calculate the difference between the tran_cam2rob * cam3D and the ground truth

# step 1
rob3D_goal = []
for idx,i in enumerate(cam3D):
    i.append(1) # add a 1 to the 3-element vector to make it as 4-element unit vector
    rob3D_goal.append(np.matmul(tran_cam2rob, np.transpose(i))[0:3]) # fetch only the first three elements

print(rob3D_goal)

# test one point
test_point = [0.09261410380182904, 0.06318547847362578, 0.898, 1]
rob3D_test = np.matmul(tran_cam2rob, np.transpose(test_point))[0:3]
print("test point in robot frame: ")
print(rob3D_test)
# step 2
errX = []
errY = []
errZ = []
dist = []
for idx, val in enumerate(rob3D):
    errX.append(val[0]-rob3D_goal[idx][0]) # x-differenceS
    errY.append(val[1]-rob3D_goal[idx][1]) # y-differenceS
    errZ.append(val[2]-rob3D_goal[idx][2]) # z-differenceS
    dist.append(np.linalg.norm(val-rob3D_goal[idx])) # absolute differene in distance

# plot results
plt.title("Calibration errors", loc = 'center', fontdict={'fontsize': 20})
plt.xlabel("Point Nr")
plt.ylabel("Diffrence in meter")
plt.plot(errX,'r', label='x-diff')
plt.plot(errY,'b', label='y-diff')
plt.plot(errZ,'g', label='z-diff')
plt.plot(dist,'m', label='dist-diff')
leg = plt.legend(loc='upper right')

plt.show()
