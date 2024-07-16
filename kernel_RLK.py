import numpy as np
def get_RLK(A, alpha=4.):
    """ Compute regularized Laplacian kernel.

    Parameters:
        A -- Adjacency matrix.
        alpha -- Diffusion parameter (positive float, 4.0 by default).

    Return:
        RLK -- Regularized Laplacian kernel matrix.
    """

    from scipy.linalg import inv

    # N is the number of vertices
    N = A.shape[0]
    for idx in range(N):
        A[idx, idx] = 0

    I = np.identity(N)
    D = np.zeros((N, N))
    for idx in range(N):
        D[idx, idx] = sum(A[idx, :])
    L = D - A
    RLK = inv(I + alpha * L)

    return RLK
