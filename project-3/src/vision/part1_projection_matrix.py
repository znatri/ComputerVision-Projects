import numpy as np


def calculate_projection_matrix(
    points_2d: np.ndarray, points_3d: np.ndarray
) -> np.ndarray:
    """
    To solve for the projection matrix. You need to set up a system of
    equations using the corresponding 2D and 3D points:

                                                      [ M11      [ u1
                                                        M12        v1
                                                        M13        .
                                                        M14        .
    [ X1 Y1 Z1 1 0  0  0  0 -u1*X1 -u1*Y1 -u1*Z1        M21        .
      0  0  0  0 X1 Y1 Z1 1 -v1*X1 -v1*Y1 -v1*Z1        M22        .
      .  .  .  . .  .  .  .    .     .      .       *   M23   =    .
      Xn Yn Zn 1 0  0  0  0 -un*Xn -un*Yn -un*Zn        M24        .
      0  0  0  0 Xn Yn Zn 1 -vn*Xn -vn*Yn -vn*Zn ]      M31        .
                                                        M32        un
                                                        M33 ]      vn ]

    Then you can solve this using least squares with np.linalg.lstsq() or SVD.
    Notice you obtain 2 equations for each corresponding 2D and 3D point
    pair. To solve this, you need at least 6 point pairs.

    Args:
        points_2d: A numpy array of shape (N, 2)
        points_3d: A numpy array of shape (N, 3)

    Returns:
        M: A numpy array of shape (3, 4) representing the projection matrix
    """
    ###########################################################################
    # TODO: YOUR CODE HERE                                                    #
    ###########################################################################

    # Number of elements
    n = points_2d.shape[0]

    # Setup input matrix
    a = np.zeros([2*n, 11])

    # Build output matrix
    b = np.reshape(points_2d, (2*n))

    # Build input matrix
    for i in range(0, n):
        X, Y, Z = points_3d[i]
        U, V = points_2d[i]
        a[i * 2] = np.array([X, Y, Z, 1, 0, 0, 0, 0, -U*X, -U*Y, -U*Z])
        a[(i * 2) + 1] = np.array([0, 0, 0, 0, X, Y, Z, 1, -V*X, -V*Y, -V*Z])

    # Projection Matrix 
    # Run linear regression and reshape matrix
    M, residuals, rank, s = np.linalg.lstsq(a, b, rcond=None)
    # Add M(3, 4) = 1
    M = np.append(M, 1.0).reshape(3, 4)

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return M


def projection(P: np.ndarray, points_3d: np.ndarray) -> np.ndarray:
    """
    Computes projection from [X,Y,Z] in non-homogenous coordinates to
    (x,y) in non-homogenous image coordinates.
    Args:
        P: 3 x 4 projection matrix
        points_3d: n x 3 array of points [X_i,Y_i,Z_i]
    Returns:
        projected_points_2d: n x 2 array of points in non-homogenous image
            coordinates
    """

    ###########################################################################
    # TODO: YOUR CODE HERE                                                    #
    ###########################################################################

    # Reshape [ X Y Z ] to [ X Y Z 1 ]
    n = points_3d.shape[0]
    points_3d = np.column_stack((points_3d, [1] * n))
    # points_3d = np.vstack([points_3d.T, np.ones(n)]).T

    uv = np.zeros([n, 3])
    for i in range(n):
        uv[i] = np.dot(P, points_3d[i])
        uv[i, 0] = np.divide(uv[i, 0], uv[i, 2])
        uv[i, 1] = np.divide(uv[i, 1], uv[i, 2])

    projected_points_2d = uv.T[:2,:].T
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return projected_points_2d


def calculate_camera_center(M: np.ndarray) -> np.ndarray:
    """
    Returns the camera center matrix for a given projection matrix.

    Args:
    -   M: A numpy array of shape (3, 4) representing the projection matrix

    Returns:
    -   cc: A numpy array of shape (1, 3) representing the camera center
            location in world coordinates
    """
    ###########################################################################
    # TODO: YOUR CODE HERE                                                    #
    ###########################################################################

    Q = M[:, :3] 
    M4 = M[:, 3]
    cc = np.dot(-np.linalg.inv(Q), M4)

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return cc
