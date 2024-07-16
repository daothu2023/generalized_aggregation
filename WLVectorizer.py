import WLGraphNodeKernel
import numpy as np
from sklearn.preprocessing import normalize
class WLVectorizer():
    def __init__(self, r=3, normalization=True):
        self.vectObject = WLGraphNodeKernel.WLGraphNodeKernel(r, normalization)
        self.normalization = normalization

    def transform(self, G_list):

        #Transform returns a sparse matrix of feature vectors not normalized
        return self.vectObject.transform(G_list)

    def getnfeatures(self):
        return self.vectObject.getnfeatures()
