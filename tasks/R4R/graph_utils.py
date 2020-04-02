# coding=utf-8
# Copyright 2019 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Utils for loading and drawing graphs of the houses."""

from __future__ import print_function

import json
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use('Agg')

import networkx as nx
import numpy as np
from numpy.linalg import norm


def load(connections_file):
    """Loads a networkx graph for a given scan.
  
    Args:
      connections_file: A string with the path to the .json file with the
        connectivity information.
    Returns:
      A networkx graph.
    """
    with open(connections_file) as f:
        lines = json.load(f)
        nodes = np.array([x['image_id'] for x in lines])
        matrix = np.array([x['unobstructed'] for x in lines])
        mask = [x['included'] for x in lines]
        matrix = matrix[mask][:, mask]
        nodes = nodes[mask]
        pos2d = {x['image_id']: np.array(x['pose'])[[3, 7]] for x in lines}
        pos3d = {x['image_id']: np.array(x['pose'])[[3, 7, 11]] for x in lines}
    
    graph = nx.from_numpy_matrix(matrix)
    graph = nx.relabel.relabel_nodes(graph, dict(enumerate(nodes)))
    nx.set_node_attributes(graph, pos2d, 'pos2d')
    nx.set_node_attributes(graph, pos3d, 'pos3d')
    
    weight2d = {(u, v): norm(pos2d[u] - pos2d[v]) for u, v in graph.edges}
    weight3d = {(u, v): norm(pos3d[u] - pos3d[v]) for u, v in graph.edges}
    nx.set_edge_attributes(graph, weight2d, 'weight2d')
    nx.set_edge_attributes(graph, weight3d, 'weight3d')
    
    return graph


def draw(graph, predicted_path, reference_path, output_filename, **kwargs):
    """Generates a plot showing the graph, predicted and reference paths.
  
    Args:
      graph: A networkx graph.
      predicted_path: A list with the ids of the nodes in the predicted path.
      reference_path: A list with the ids of the nodes in the reference path.
      output_filename: A string with the path where to store the generated image.
      **kwargs: Key-word arguments for aesthetic control.
    """
    plt.figure(figsize=(10, 10))
    ax = plt.gca()
    pos = nx.get_node_attributes(graph, 'pos2d')
    
    # Zoom in.
    # xs = [pos[node][0] for node in predicted_path + reference_path]
    # ys = [pos[node][1] for node in predicted_path + reference_path]
    # min_x, max_x, min_y, max_y = min(xs), max(xs), min(ys), max(ys)
    # center_x, center_y = (min_x + max_x) / 2, (min_y + max_y) / 2
    # zoom_margin = kwargs.get('zoom_margin', 1.3)
    # max_range = zoom_margin * max(max_x - min_x, max_y - min_y)
    # half_range = max_range / 2
    # ax.set_xlim(center_x - half_range, center_x + half_range)
    # ax.set_ylim(center_y - half_range, center_y + half_range)
    
    # Background graph.
    nx.draw(graph,
            pos,
            edge_color=kwargs.get('background_edge_color', 'lightgrey'),
            node_color=kwargs.get('background_node_color', 'lightgrey'),
            node_size=kwargs.get('background_node_size', 60),
            width=kwargs.get('background_edge_width', 0.5))
    
    # Prediction graph.
    predicted_path_graph = nx.DiGraph()
    predicted_path_graph.add_nodes_from(predicted_path)
    predicted_path_graph.add_edges_from(
        zip(predicted_path[:-1], predicted_path[1:]))
    nx.draw(predicted_path_graph,
            pos,
            arrowsize=kwargs.get('prediction_arrowsize', 15),
            edge_color=kwargs.get('prediction_edge_color', 'red'),
            node_color=kwargs.get('prediction_node_color', 'red'),
            node_size=kwargs.get('prediction_node_size', 130),
            width=kwargs.get('prediction_edge_width', 2.0),
            # with_labels=True,
            )
    
    # Reference graph.
    # reference_path_graph = nx.DiGraph()
    # reference_path_graph.add_nodes_from(reference_path)
    # reference_path_graph.add_edges_from(
    #     zip(reference_path[:-1], reference_path[1:]))
    # nx.draw(reference_path_graph,
    #         pos,
    #         # node_color=range(24),
    #         arrowsize=kwargs.get('reference_arrowsize', 15),
    #         edge_color=kwargs.get('reference_edge_color', 'dodgerblue'),
    #         node_color=kwargs.get('reference_node_color', 'dodgerblue'),
    #         node_size=kwargs.get('reference_node_size', 130),
    #         width=kwargs.get('reference_edge_width', 2.0))
    
    # Intersection graph.
    # intersection_path_graph = nx.DiGraph()
    # common_nodes = set(predicted_path_graph.nodes.keys()).intersection(
    #     set(reference_path_graph.nodes.keys()))
    # intersection_path_graph.add_nodes_from(common_nodes)
    # common_edges = set(predicted_path_graph.edges.keys()).intersection(
    #     set(reference_path_graph.edges.keys()))
    # intersection_path_graph.add_edges_from(common_edges)
    # nx.draw(intersection_path_graph,
    #         pos,
    #         arrowsize=kwargs.get('intersection_arrowsize', 15),
    #         edge_color=kwargs.get('intersection_edge_color', 'limegreen'),
    #         node_color=kwargs.get('intersection_node_color', 'limegreen'),
    #         node_size=kwargs.get('intersection_node_size', 130),
    #         width=kwargs.get('intersection_edge_width', 2.0))
    
    # # Start graph.
    start_graph = nx.DiGraph()
    start_graph.add_nodes_from([reference_path[0]])
    nx.draw(start_graph,
            pos,
            node_color=kwargs.get('start_node_color', 'yellow'),
            node_size=kwargs.get('start_node_size', 150))
    
    # End graph.
    end_graph = nx.DiGraph()
    end_graph.add_nodes_from([reference_path[-1]])
    nx.draw(end_graph,
            pos,
            node_shape='*',
            node_color=kwargs.get('end_node_color', 'yellow'),
            node_size=kwargs.get('end_node_size', 150))
    
    plt.savefig(output_filename)
    plt.show()
    plt.close()


if __name__ == "__main__":
    # connections_file = "../../simulator/connectivity/X7HyMhZNoso_connectivity.json"
    # predicted_path = ["5445d1e47e204f598d836d7940013231",
    #                   "997ec56720304a069672a8a0fe2b80e6",
    #                   "0990cc040127481d97727123df0c9e56",
    #                   "2739968bfacb412cb7997d6d59f461c2",
    #                   "d0468b580cd146778bae5e486bf3e50e",
    #                   "321dba26c17b464f9623c86917babfeb",
    #                   "29a55ded3f094585a9bf7c8d2bf312be",
    #                   "32a001fd02e24ca88268228786d22bef",
    #                   "3c90780cab6b4495a89cbf1dac752255",
    #                   "273d144230a740888de22539802509ad",
    #                   "87b4508bfdbf497299bd26eb4b23282a",
    #                   "6207c0c642ec4cdf95a41a9cc0b7fb38",
    #                   "a39b7d7481364573bb7da6ee462aab6d",
    #                   "baff9dfe176543a88ec8f09a99202054",
    #                   "9716d6efd8004e66b1ca7faf7ad86438",
    #                   "e739cfd915d642b4bd23743e15d1480b",
    #                   "89eaa0c1aaf4471f9d081d6cc358cc44",
    #                   "c0e590504b61489fba3e0c2a12664a26",
    #                   "e40ff9839f0b4a35a25e9dc16d391ae4",
    #                   "0e0c08b705704f80b5f31c2bd3a40583",
    #                   "cd608227f6c94b91af3db8bf6cd28abd",
    #                   "ace68ede9cfe44d3842cfe5d937cfa36",
    #                   "987fd31155514f6facb131bd5c14881d"]
    # reference_path = ["5445d1e47e204f598d836d7940013231",
    #                   "997ec56720304a069672a8a0fe2b80e6",
    #                   "0990cc040127481d97727123df0c9e56",
    #                   "2739968bfacb412cb7997d6d59f461c2",
    #                   "d0468b580cd146778bae5e486bf3e50e",
    #                   "321dba26c17b464f9623c86917babfeb",
    #                   "29a55ded3f094585a9bf7c8d2bf312be",
    #                   "32a001fd02e24ca88268228786d22bef",
    #                   "3c90780cab6b4495a89cbf1dac752255",
    #                   "273d144230a740888de22539802509ad",
    #                   "87b4508bfdbf497299bd26eb4b23282a",
    #                   "6207c0c642ec4cdf95a41a9cc0b7fb38",
    #                   "a39b7d7481364573bb7da6ee462aab6d",
    #                   "baff9dfe176543a88ec8f09a99202054",
    #                   "9716d6efd8004e66b1ca7faf7ad86438",
    #                   "e739cfd915d642b4bd23743e15d1480b",
    #                   "89eaa0c1aaf4471f9d081d6cc358cc44",
    #                   "c0e590504b61489fba3e0c2a12664a26",
    #                   "e40ff9839f0b4a35a25e9dc16d391ae4",
    #                   "0e0c08b705704f80b5f31c2bd3a40583",
    #                   "cd608227f6c94b91af3db8bf6cd28abd",
    #                   "ace68ede9cfe44d3842cfe5d937cfa36",
    #                   "987fd31155514f6facb131bd5c14881d"]
    # output_file = "try.png"
    # graph = load(connections_file)
    # draw(graph, predicted_path, reference_path, output_file)
    
    split = 6
    nbs_file = "../vln_visualize/plot/json/follower_nbs_sample2_R" + str(split) + "R_val_iter_0_val_unseen.json"
    rcm_file = "../vln_visualize/plot/json/follower_rcm_sample2_R" + str(split) + "R_val_iter_0_val_unseen.json"
    sf_file = "../vln_visualize/plot/json/follower_sf_sample2_R" + str(split) + "R_val_iter_0_val_unseen.json"
    seq_file = "../vln_visualize/plot/json/follower_seq_sample2_R" + str(split) + "R_val_iter_0_val_unseen.json"
    # map_file = "../hierarchical_vln/cvpr/plot/json/follower_map_nbs2rcm.json"
    gt_output_file = "../vln_assets/data/r" + str(split) + "r_sample2_traj/gt"
    nbs_output_file = "../vln_assets/data/r" + str(split) + "r_sample2_traj/nbs"
    rcm_output_file = "../vln_assets/data/r" + str(split) + "r_sample2_traj/rcm"
    sf_output_file = "../vln_assets/data/r" + str(split) + "r_sample2_traj/sf"
    seq_output_file = "../vln_assets/data/r" + str(split) + "r_sample2_traj/seq"
    graphs = {}
    with open(rcm_file, 'r') as f:
        rcm_data = json.load(f)
    with open(sf_file, 'r') as f:
        sf_data = json.load(f)
    with open(seq_file, 'r') as f:
        seq_data = json.load(f)
    with open(nbs_file, 'r') as f:
        nbs_data = json.load(f)
    # with open(map_file, 'r') as f:
    #     map_dict = json.load(f)
    count = 0
    start = 1200
    end = start + 100
    for id, d in rcm_data.items():
        # if int(id.split('_')[0]) < start or int(id.split('_')[0]) >= end:
        #     continue
        count += 1
        if count < start or count >= end:
            continue
        if d['scan'] not in graphs:
            connections_file = "simulator/connectivity/{}_connectivity.json".format(d['scan'])
            graphs[d['scan']] = load(connections_file)
        output_file = rcm_output_file + "/rcm_{}.png".format(id)
        draw(graphs[d['scan']], d['trajectory'], d['expert_trajectory'], output_file)
        # if count >= 1000:
        #     break
    for id, d in sf_data.items():
        # if int(id.split('_')[0]) < start or int(id.split('_')[0]) >= end:
        #     continue
        count += 1
        if count < start or count >= end:
            continue
        if d['scan'] not in graphs:
            connections_file = "simulator/connectivity/{}_connectivity.json".format(d['scan'])
            graphs[d['scan']] = load(connections_file)
        output_file = sf_output_file + "/sf_{}.png".format(id)
        draw(graphs[d['scan']], d['trajectory'], d['expert_trajectory'], output_file)
        # if count >= 1000:
        #     break
    for id, d in seq_data.items():
        # if int(id.split('_')[0]) < start or int(id.split('_')[0]) >= end:
        #     continue
        count += 1
        if count < start or count >= end:
            continue
        if d['scan'] not in graphs:
            connections_file = "simulator/connectivity/{}_connectivity.json".format(d['scan'])
            graphs[d['scan']] = load(connections_file)
        output_file = seq_output_file + "/seq_{}.png".format(id)
        draw(graphs[d['scan']], d['trajectory'], d['expert_trajectory'], output_file)
        # if count >= 1000:
        #     break
    for id, d in nbs_data.items():
        # id = map_dict[id]
        # if int(id.split('_')[0]) < start or int(id.split('_')[0]) >= end:
        #     continue
        count += 1
        if count < start or count >= end:
            continue
        if d['scan'] not in graphs:
            connections_file = "simulator/connectivity/{}_connectivity.json".format(d['scan'])
            graphs[d['scan']] = load(connections_file)
        output_file = nbs_output_file + "/nbs_{}.png".format(id)
        draw(graphs[d['scan']], d['trajectory'], d['expert_trajectory'], output_file,
             prediction_edge_color='limegreen', prediction_node_color='limegreen')
        output_file = gt_output_file + "/gt_{}.png".format(id)
        draw(graphs[d['scan']], d['expert_trajectory'], d['expert_trajectory'], output_file,
             prediction_edge_color='dodgerblue', prediction_node_color='dodgerblue')
        # if count >= 1000:
        #     break
    
    #  for id, d in rcm_data.items():
    #     if int(id.split('_')[0]) < 12730 or int(id.split('_')[0]) >= 12750:
    #         continue
    #     if d['scan'] not in graphs:
    #         connections_file = "simulator/connectivity/{}_connectivity.json".format(d['scan'])
    #         graphs[d['scan']] = load(connections_file)
    #     output_file = rcm_output_file + "/rcm_{}.png".format(id)
    #     draw(graphs[d['scan']], d['trajectory'], d['expert_trajectory'], output_file)
    #     count += 1
    #     # if count >= 1000:
    #     #     break
    # for id, d in sf_data.items():
    #     if int(id.split('_')[0]) < 12730 or int(id.split('_')[0]) >= 12750:
    #         continue
    #     if d['scan'] not in graphs:
    #         connections_file = "simulator/connectivity/{}_connectivity.json".format(d['scan'])
    #         graphs[d['scan']] = load(connections_file)
    #     output_file = sf_output_file + "/sf_{}.png".format(id)
    #     draw(graphs[d['scan']], d['trajectory'], d['expert_trajectory'], output_file)
    #     count += 1
    #     # if count >= 1000:
    #     #     break
    # for id, d in seq_data.items():
    #     if int(id.split('_')[0]) < 12730 or int(id.split('_')[0]) >= 12750:
    #         continue
    #     if d['scan'] not in graphs:
    #         connections_file = "simulator/connectivity/{}_connectivity.json".format(d['scan'])
    #         graphs[d['scan']] = load(connections_file)
    #     output_file = seq_output_file + "/seq_{}.png".format(id)
    #     draw(graphs[d['scan']], d['trajectory'], d['expert_trajectory'], output_file)
    #     count += 1
    #     # if count >= 1000:
    #     #     break
    # for id, d in nbs_data.items():
    #     id = map_dict[id]
    #     if int(id.split('_')[0]) < 12730 or int(id.split('_')[0]) >= 12750:
    #         continue
    #     if d['scan'] not in graphs:
    #         connections_file = "simulator/connectivity/{}_connectivity.json".format(d['scan'])
    #         graphs[d['scan']] = load(connections_file)
    #     output_file = nbs_output_file + "/nbs_{}.png".format(id)
    #     draw(graphs[d['scan']], d['trajectory'], d['expert_trajectory'], output_file)
    #     count += 1
    #     # if count >= 1000:
    #     #     break
