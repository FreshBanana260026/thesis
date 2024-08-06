#!/usr/bin/python
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import copy
# DESCRIPTION:
#


class RigidTransFitter3D(object):

    def __init__(self):
        pass

    def get_transform(self, A_input, B_input):

        # Input format of A_input, B_input:
        # P = [[x0,y0,z0],[x1,y1,z1],...,[xn,yn,zn]]

        # Output format:
        # numpy matrices for rotation and translation
        # Rt, t


        # A_input and B_input should have same length
        assert len(A_input) == len(B_input)

        # Reorganize format of A_input, B_input:
        # P = [[x1, x2, x3, ... , xn],
        #      [y1, y2, y3, ... , yn]
        #      [z1, z2, z3, ... , zn]]
        A = np.transpose(np.mat(A_input))
        B = np.transpose(np.mat(B_input))

        # check if A and B have correct format
        num_rows, num_cols = A.shape;
        if num_rows != 3:
            raise Exception("matrix A is not 3xN, it is {}x{}".format(num_rows, num_cols))
        num_rows, num_cols = B.shape;
        if num_rows != 3:
            raise Exception("matrix B is not 3xN, it is {}x{}".format(num_rows, num_cols))

        # find mean value of x, y, z separately
        # centroid_P = [<x_P>,<y_P>,<z_P>]
        centroid_A = np.mean(A, axis=1)
        centroid_B = np.mean(B, axis=1)

        # subtract mean
        Am = A - np.tile(centroid_A, (1, num_cols))
        Bm = B - np.tile(centroid_B, (1, num_cols))

        # find 'correlation' matrix (proportional to covariance matrix of A and B)
        H = Am * np.transpose(Bm)

        # sanity check
        #if linalg.matrix_rank(H) < 3:
        #    raise ValueError("rank of H = {}, expecting 3".format(linalg.matrix_rank(H)))

        # find rotation via singular value decomposition (svd) of correlation matrix
        U, S, Vt = np.linalg.svd(H)
        R = Vt.T * U.T

        # special reflection case
        if np.linalg.det(R) < 0:
            print("det(R) < R, reflection detected!, correcting for it ...\n");
            Vt[2,:] *= -1
            R = Vt.T * U.T

        # find translation
        t = -R*centroid_A + centroid_B

        tran = np.zeros((4,4))
        tran[0:3,0:3] = R
        tran[0:3,3] = np.resize(t, (1, 3))
        tran[3,3] = 1
        
        # return rotation and translation matrices
        return tran





if __name__=='__main__':

    # Test with random data

    # Random rotation and translation
    R = np.mat(np.random.rand(3,3))
    t = np.mat(np.random.rand(3,1))

    # make R a proper rotation matrix, force orthonormal
    U, S, Vt = np.linalg.svd(R)
    R = U*Vt

    # remove reflection
    if np.linalg.det(R) < 0:
        Vt[2,:] *= -1
        R = U*Vt

    # number of points
    n = 10
    A = np.mat(np.random.rand(3, n))
    B = R*A + np.tile(t, (1, n))
    A = A.transpose().tolist()
    B = B.transpose().tolist()
    # transform to np.array as this is what the function expects
    A_arr = np.array(A)
    B_arr = np.array(B)

    # Recover R and t
    Transformer = RigidTransFitter3D()
    trans = Transformer.get_transform(A_arr, B_arr)

    B2 = []
    A_temp = copy.deepcopy(A)
    for idx, i in enumerate(A_temp):
        print(i)
        i.append(1) # add a 1 to the 3-element vector to make it as 4-element unit vector
        B2.append(np.matmul(trans, np.transpose(i))[0:3]) # fetch only the first three elements
    # Compare the recovered R and t with the original
    #B2 = (ret_R*A) + np.tile(ret_t, (1, n))

    # Find the root mean squared error
    #err = B2 - B
    #err = np.multiply(err, err)
    #err = sum(err)
    #rmse = np.sqrt(err/n)
    # Remake A, B, B3 to np.array for plotting
    A = np.array(A).transpose()
    B = np.array(B).transpose()
    B2 = np.array(B2).transpose()
    
    print(B2.shape)
    print(B.shape)
    print(A.shape)

    fig = plt.figure()
    fig2 = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax2 = fig2.add_subplot(111, projection='3d')

    # Original Points (A and B)
    ax.scatter(A[0,:], A[1,:], A[2,:], c='r', marker='o', label='Original A')
    ax.scatter(B[0,:], B[1,:], B[2,:], c='g', marker='^', label='Original B')

    # Transformed Points (B2) with desired points B
    ax2.scatter(B2[0,:], B2[1,:], B2[2,:], c='r', marker='x', label='Transformed A')
    ax2.scatter(B[0,:], B[1,:], B[2,:], c='g', marker='*', label='Original B')

    ax.set_title('Original Points')
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    ax.legend()

    ax2.set_title('Transformed Points')
    ax2.set_xlabel('X Label')
    ax2.set_ylabel('Y Label')
    ax2.set_zlabel('Z Label')
    ax2.legend()
    plt.show()

    # print("Points A")
    # print(A)
    # print("")

    # print("Points B")
    # print(B)
    # print("")

    # print("Ground truth rotation")
    # print(R)

    # print("Recovered rotation")
    # print(ret_R)
    # print("")

    # print("Ground truth translation")
    # print(t)

    # print("Recovered translation")
    # print(ret_t)
    # print("")

    # print("RMSE:", rmse)

    # if np.all(rmse < 1e-5):
    #     print("Everything looks good!\n");
    # else:
    #     print("Hmm something doesn't look right ...\n");
