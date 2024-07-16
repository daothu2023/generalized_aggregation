import numpy as np
import networkx as nx
import copy
from KernelTools import new_convert_to_sparse_matrix
import math
from scipy.sparse import dok_matrix
from sklearn.preprocessing import normalize


class WLGraphNodeKernel(): #GraphKernel
    def __init__(self, r=1, normalization=False):
        self.h = r
        self.normalization = normalization
        self.__startsymbol = '!' #special symbols used in encoding
        self.__conjsymbol = '#'
        self.__endsymbol = '?'
        self.__fsfeatsymbol = '*'
        self.__version = 0
        self.__contextsymbol = '@'

    def kernelFunction(self, g_1, g_2):

        gl = [g_1, g_2]
        return self.computeGram(gl)[0, 1]



    def transform(self, graph_list):

        MapEncToId = None

        n = len(graph_list)  # number of graphs
        phi = [[{} for i in range(n)] for j in range(self.h+1)]
        NodeIdToLabelId = [0] * n  # NodeIdToLabelId[i][j] is labelid of node j in graph i
        label_lookup = {}  # map from features to corresponding id
        label_counter = 0  # incremental value for label ids

        for i in range(n):  # for each graph
            NodeIdToLabelId[i] = {}

            for idx_j, j in enumerate(graph_list[i].graph['node_order']):
                if graph_list[i].nodes[j]['label'] not in label_lookup:#update label_lookup and label ids from first iteration that consider node's labels
                    label_lookup[graph_list[i].nodes[j]['label']] = label_counter
                    NodeIdToLabelId[i][j] = label_counter
                    label_counter += 1
                else:
                    NodeIdToLabelId[i][j] = label_lookup[graph_list[i].nodes[j]['label']]
                # print self.__fsfeatsymbol
                feature = label_lookup[graph_list[i].nodes[j]['label']]
                if idx_j not in phi[0][i]:
                     phi[0][i][idx_j] = {}
                if feature not in phi[0][i][idx_j]:
                     phi[0][i][idx_j][feature] = 0.0
                phi[0][i][idx_j][feature] += 1.0

        # MAIN LOOP
        # TOFO generate a vector for each it value
        it = 1
        NewNodeIdToLabelId = copy.deepcopy(NodeIdToLabelId)  # labels id of nex iteration

        # test in order to generate different features for different r
        # label_lookup = {}

        while it <= self.h:  # each iteration compute the next labellings (that are contexts of the previous)
            # label_lookup = {}

            for i in range(n):  # for each graph
                for idx_j, j in enumerate(graph_list[i].graph['node_order']):  # for each node, consider its neighbourhood
                    neighbors = []
                    for u in graph_list[i].neighbors(j):
                        neighbors.append(NodeIdToLabelId[i][u])
                    neighbors.sort()  # sorting neighbours

                    long_label_string = self.__fsfeatsymbol+str(NodeIdToLabelId[i][j])+self.__startsymbol  # compute new labels id
                    for u in neighbors:
                        long_label_string += str(u)+self.__conjsymbol
                    long_label_string = long_label_string[:-1]+self.__endsymbol

                    if long_label_string not in label_lookup:
                        label_lookup[long_label_string] = label_counter
                        NewNodeIdToLabelId[i][j] = label_counter
                        # print label_counter
                        label_counter += 1
                    else:
                        NewNodeIdToLabelId[i][j] = label_lookup[long_label_string]

                    feature = NewNodeIdToLabelId[i][j]
                    if idx_j not in phi[it][i]:
                        phi[it][i][idx_j]={}
                    if feature not in phi[it][i][idx_j]:
                        phi[it][i][idx_j][feature] = 0.0
                    phi[it][i][idx_j][feature] += 1.0


            NodeIdToLabelId = copy.deepcopy(NewNodeIdToLabelId)  # update current labels id
            it = it + 1


        ve = phi

        # df = DataFrame(phi[j][i],).T.fillna(0)

        ve = [[new_convert_to_sparse_matrix(phi[j][i], label_counter) for i in range(n)] for j in range(self.h+1)]


        return ve

    def computeKernelMatrixTrain(self, Graphs):
        return self.computeGram(Graphs)
    def computeGram(self, g_it, precomputed=None):
        if precomputed is None:
            precomputed = self.transform(g_it)
        return precomputed.dot(precomputed.T).todense().tolist()


