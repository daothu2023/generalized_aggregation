
from eden.graph import Vectorizer
from sklearn.metrics import pairwise

def NSPDK(g, d, r):
    """ Vectorize graph nodes

    Needs graph input

    Return: a matrix in which rows are the vectors that represents for nodes
    """

    vec = Vectorizer(nbits=16,
                     discrete=True,
                     d=d,
                     r=r,
                     )

    M = vec.vertex_transform(g)
    M = M[0]
    K = pairwise.linear_kernel(M, M)

    return K
