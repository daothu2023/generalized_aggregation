from numpy.linalg import norm
def instance_to_graph(input = None,dict_labels={}, counter=[1]):
    """
    Function that reads a graph dataset encoded in gspan format from an inpout stream.

    """
    import gspan
    return gspan.gspan_to_eden(input,dict_labels,counter)



def setHashSubtreeIdentifierBigDAG(T, nodeID, sep='|',labels=True,veclabels=True):
    setOrder(T, nodeID, labels=labels,veclabels=veclabels)
    if 'subtreeID' in T.node[nodeID]:
        return T.node[nodeID]['subtreeID']
    stri = str(T.node[nodeID]['label'])+str(T.node[nodeID]['veclabel'])
    if stri.find(sep) != -1:
        print("ERROR: identifier " + sep + "used in label. Please set it with setHashSep(newsep)")
    for c in T.node[nodeID]['childrenOrder']:
        stri += sep + setHashSubtreeIdentifierBigDAG(T,c,sep,labels,veclabels)
    T.node[nodeID]['subtreeID'] = str(stri)
    return T.node[nodeID]['subtreeID']

def setOrder(T, nodeID, sep='|',labels=True, veclabels = True, order = "norm"):
    if veclabels:
        return setOrderVeclabels(T, nodeID, sep, labels, order)
    else:
       return setOrderNoVeclabels(T, nodeID, sep, labels)


def setOrderVeclabels(T, nodeID, sep, labels, order):
        """
        The method computes an identifier of the node based on
        1) the label of the node self
        2) the hash values of the children of self
        For each visited node the hash value is stored into the attribute subtreeId
        The label and the identifiers of the children nodes are separated by the char 'sep'
        """
        if 'orderString' in T.node[nodeID]:
            return T.node[nodeID]['orderString']
        stri = str(T.node[nodeID]['label'])

        if stri.find(sep) != -1:
            print("ERROR: identifier " + sep + "used in label. Please set it with setHashSep(newsep)")
        succ_labels = []
        if len(T.successors(nodeID)) > 0:
            stri += sep+str(len(T.successors(nodeID)))
        for c in T.successors(nodeID):
            if order == 'gaussian':
                dist = gaussianKernel(T.node[nodeID]['veclabel'], T.node[c]['veclabel'], 1.0/len(T.node[c]['veclabel']))#self.beta)
            elif order == 'norm':
                dist = norm(T.node[nodeID]['veclabel'])
            else:
                print("no ordering specified")
            tup = ([setOrderVeclabels(T, c, sep, labels, order), dist], c)
            succ_labels.append(tup)
        succ_labels.sort(key=lambda x: (x[0][0], x[0][1]))#cmp = lambda x, y: cmp(x[0], y[0])
        children = []
        for l in succ_labels:
            stri += sep + str(l[0][0])
            children.append(l[1])
        T.node[nodeID]['orderString'] = stri
        T.node[nodeID]['childrenOrder']= children
        return T.node[nodeID]['orderString']