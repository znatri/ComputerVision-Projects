"""Fundamental matrix utilities."""

import numpy as np


def normalize_points(points: np.ndarray) -> (np.ndarray, np.ndarray):
    """
    Perform coordinate normalization through linear transformations.
    Args:
        points: A numpy array of shape (N, 2) representing the 2D points in
            the image

    Returns:
        points_normalized: A numpy array of shape (N, 2) representing the
            normalized 2D points in the image
        T: transformation matrix representing the product of the scale and
            offset matrices
    """
    ###########################################################################
    # TODO: YOUR CODE HERE                                                    #
    ###########################################################################
    n = points.shape[0]
    points_normalized = np.zeros([n, 3])

    # Offsets cu and cv are the mean coordinates
    c = np.mean(points, axis=0) # [c_u, c_v]

    # To compute a scale s you could estimate the standard deviation after subtracting the means. 
    # Then the scale factor s would be the reciprocal of whatever estimate of the scale you are using.
    s = np.reciprocal(np.std(points-c, axis=0))

    offset_m = np.zeros([3, 3])
    np.fill_diagonal(offset_m, 1)
    offset_m[:, 2] = np.hstack([-c, 1]) 

    scale_m = np.zeros([3, 3])
    np.fill_diagonal(scale_m, np.hstack([s, 1]))

    T = np.dot(scale_m, offset_m)

    points = np.column_stack((points, [1] * n))
    # points = np.vstack([points.T, np.ones(n)]).T
    points_normalized = T.dot(points.T).T[:, :2]

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return points_normalized, T


def unnormalize_F(F_norm: np.ndarray, T_a: np.ndarray, T_b: np.ndarray) -> np.ndarray:
    """
    Adjusts F to account for normalized coordinates by using the transformation
    matrices.

    Args:
        F_norm: A numpy array of shape (3, 3) representing the normalized
            fundamental matrix
        T_a: Transformation matrix for image A
        T_B: Transformation matrix for image B

    Returns:
        F_orig: A numpy array of shape (3, 3) representing the original
            fundamental matrix
    """
    ###########################################################################
    # TODO: YOUR CODE HERE                                                    #
    ###########################################################################

    F_orig = np.matmul(np.matmul(T_b.T, F_norm), T_a)

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return F_orig


def estimate_fundamental_matrix(
    points_a: np.ndarray, points_b: np.ndarray
) -> np.ndarray:
    """
    Calculates the fundamental matrix. You may use the normalize_points() and
    unnormalize_F() functions here.

    Args:
        points_a: A numpy array of shape (N, 2) representing the 2D points in
            image A
        points_b: A numpy array of shape (N, 2) representing the 2D points in
            image B

    Returns:
        F: A numpy array of shape (3, 3) representing the fundamental matrix
    """

    assert points_a.shape[0] == points_b.shape[0]

    ###########################################################################
    # TODO: YOUR CODE HERE                                                    #
    ###########################################################################
    n = points_b.shape[0]
    F = np.zeros([3, 3])

    # points_b = np.vstack([points_b.T,  np.ones(n)])
    points_norm_a, T_a = normalize_points(points_a)
    points_norm_b, T_b = normalize_points(points_b)

    # Setup input matrix
    a = np.zeros([n, 8])

    # Setup output matrix
    b = -np.ones([n, 1])

    # Build input regression matrix
    for i in range(0, n):
        u_1, v_1 = points_norm_a[i]
        u_2, v_2 = points_norm_b[i]
        a[i] = np.array([np.multiply(u_1, u_2), np.multiply(v_1, u_2), u_2, np.multiply(u_1, v_2), np.multiply(v_1, v_2), v_2, u_1, v_1])

    # Get Least-square solution
    F, residuals, rank, s = np.linalg.lstsq(a, b, rcond=None)  
    F = np.vstack([F, 1.0]).reshape(3,3)                     
    F = np.matmul(np.matmul(T_b.T, F), T_a)                             

    # Convert Full rank to 2 rank
    u, z, v = np.linalg.svd(F)
    z = np.array([[z[0], 0, 0],[0, z[1], 0],[0, 0, 0]])
    F = np.dot(u, z).dot(v)
    
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return F
