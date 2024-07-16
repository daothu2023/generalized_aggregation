import ioskgraph
import json
import networkx as nx


def gspan_to_eden(infile,dict_labels={}, counter=[1]):
    """
    Takes a string list in the extended gSpan format and yields networkx graphs.

    Parameters
    ----------
    input : string
        A pointer to the data source.

    """

    string_list = []
    for line in ioskgraph.read(infile):
        if line.strip():
            if line[0] in ['g', 't']:
                if string_list:
                    yield _gspan_to_networkx(string_list,dict_labels,counter)
                string_list = []
            string_list += [line]

    if string_list:
        yield _gspan_to_networkx(string_list, dict_labels, counter)


"""def _gspan_to_networkx(string_list,dict_labels={}, counter=[1]):
    graph = nx.Graph()
    graph.graph['ordered']=False
    node_order = []

    for line in string_list:
        if line.strip():
            line_list = line.split()
            firstcharacter = line_list[0]
            if firstcharacter in ['v', 'V']:
                vid = int(line_list[1])
                vlabel = line_list[2]
                if vlabel not in dict_labels:
                    dict_labels[vlabel] = counter[0]
                    counter[0] += 1
                if firstcharacter == 'v':
                    weight = 1
                else:
                    weight = 0.1
                graph.add_node(vid, label=vlabel, weight=weight, viewpoint=True)
                node_order.append(vid)

                if vlabel[0] == '^':
                    graph.node[vid]['nesting'] = True
                attribute_str = ' '.join(line_list[3:])
                if attribute_str.strip():
                    attribute_dict = json.loads(attribute_str)
                    graph.node[vid].update(attribute_dict)
            if firstcharacter == 'e':
                srcid = int(line_list[1])
                destid = int(line_list[2])
                elabel = line_list[3]
                graph.add_edge(srcid, destid, label=elabel)
                attribute_str = ' '.join(line_list[4:])
                if attribute_str.strip():
                    attribute_dict = json.loads(attribute_str)
                    graph.edge[srcid][destid].update(attribute_dict)
    graph.graph['node_order'] = node_order

    assert(len(graph) > 0), 'ERROR: generated empty graph. Perhaps wrong format?'
    return graph"""


"""def _gspan_to_networkx(string_list, dict_labels={}, counter=[1]):
    graph = nx.Graph()
    graph.graph['ordered'] = False
    node_order = []

    for line in string_list:
        if line.strip():
            line_list = line.split()
            firstcharacter = line_list[0]
            if firstcharacter in ['v', 'V']:

                vid = int(line_list[1])
                vlabel = line_list[2]
                if vlabel not in dict_labels:
                    dict_labels[vlabel] = counter[0]
                    counter[0] += 1
                if firstcharacter == 'v':
                    weight = 1
                else:
                    weight = 0.1
                graph.add_node(vid, label=vlabel, weight=weight, viewpoint=True)
                node_order.append(vid)

                if vlabel[0] == '^':
                    graph.node[vid]['nesting'] = True
                attribute_str = ' '.join(line_list[3:])
                if attribute_str.strip():
                    attribute_dict = json.loads(attribute_str)
                    graph.nodes[vid].update(attribute_dict)
            if firstcharacter == 'e':
                srcid = int(line_list[1])
                destid = int(line_list[2])
                elabel = line_list[3]
                graph.add_edge(srcid, destid, label=elabel)
                attribute_str = ' '.join(line_list[4:])
                if attribute_str.strip():
                    attribute_dict = json.loads(attribute_str)
                    graph.edge[srcid][destid].update(attribute_dict)

    if len(graph) == 0:
        print("WARNING: Generated empty graph. Perhaps wrong format?")
    else:
        graph.graph['node_order'] = node_order
        return graph"""
def _gspan_to_networkx(string_list, dict_labels={}, counter=[1]):


    graph = nx.Graph()
    graph.graph['ordered'] = False
    node_order = []

    for i, line in enumerate(string_list):
        if line.strip():
            line_list = line.split()
            firstcharacter = line_list[0]
            if firstcharacter in ['v', 'V']:
                vid = int(line_list[1])
                vlabel = line_list[2]
                if vlabel not in dict_labels:
                    dict_labels[vlabel] = counter[0]
                    counter[0] += 1
                if firstcharacter == 'v':
                    weight = 1
                else:
                    weight = 0.1
                graph.add_node(vid, label=vlabel, weight=weight, viewpoint=True)
                node_order.append(vid)

                if vlabel[0] == '^':
                    graph.nodes[vid]['nesting'] = True
                attribute_str = ' '.join(line_list[3:])
                if attribute_str.strip():
                    attribute_dict = json.loads(attribute_str)
                    graph.nodes[vid].update(attribute_dict)
            if firstcharacter == 'e':
                srcid = int(line_list[1])
                destid = int(line_list[2])
                elabel = line_list[3]
                graph.add_edge(srcid, destid, label=elabel)
                attribute_str = ' '.join(line_list[4:])
                if attribute_str.strip():
                    attribute_dict = json.loads(attribute_str)
                    graph.edge[srcid][destid].update(attribute_dict)

    if len(graph) == 0:
        print(f"WARNING: Generated empty graph at index {i}. Perhaps wrong format?")
    else:
        graph.graph['node_order'] = node_order
    return graph


