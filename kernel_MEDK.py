
import numpy as np
def get_MEDK(A, beta=0.04):
    """ Compute Markov exponential diffusion kernel.

    Parameters:
        A -- Adjacency matrix.
        beta -- Diffusion parameter (positive float, 0.04 by default).

    Return:
        MEDK -- Markov exponential diffusion kernel matrix.
    """

    from cvxopt import matrix
    from scipy.linalg import expm

    # N is the number of vertices
    N = A.shape[0]
    for idx in range(N):
        A[idx, idx] = 0
    A = matrix(A)
    D = np.zeros((N, N))
    for idx in range(N):
        D[idx, idx] = sum(A[idx, :])
    I = np.identity(N)
    M = (beta / N) * (N * I - D + A)
    MEDK = expm(M)

    return MEDK