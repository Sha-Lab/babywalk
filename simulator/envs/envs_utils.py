''' Utils for the environments '''

import sys
import json
import numpy as np
import networkx as nx
import base64


def load_nav_graphs(scans):
  ''' Load connectivity graph for each scan '''
  
  def distance(pose1, pose2):
    ''' Euclidean distance between two graph poses '''
    return ((pose1['pose'][3] - pose2['pose'][3]) ** 2
            + (pose1['pose'][7] - pose2['pose'][7]) ** 2
            + (pose1['pose'][11] - pose2['pose'][11]) ** 2) ** 0.5
  
  graphs = {}
  for scan in scans:
    with open('simulator/connectivity/%s_connectivity.json' % scan) as f:
      G = nx.Graph()
      positions = {}
      data = json.load(f)
      for i, item in enumerate(data):
        if item['included']:
          for j, conn in enumerate(item['unobstructed']):
            if conn and data[j]['included']:
              positions[item['image_id']] = np.array([item['pose'][3],
                                                      item['pose'][7],
                                                      item['pose'][11]])
              assert data[j]['unobstructed'][i], 'Graph should be undirected'
              G.add_edge(item['image_id'], data[j]['image_id'],
                         weight=distance(item, data[j]))
      nx.set_node_attributes(G, values=positions, name='position')
      graphs[scan] = G
  return graphs


def decode_base64(string):
  if sys.version_info[0] == 2:
    return base64.decodestring(bytearray(string))
  elif sys.version_info[0] == 3:
    return base64.decodebytes(bytearray(string, 'utf-8'))
  else:
    raise ValueError("decode_base64 can't handle python version {}".format(
      sys.version_info[0]))


def structured_map(function, *args, **kwargs):
  nested = kwargs.get('nested', False)
  acc = []
  for t in zip(*args):
    if nested:
      mapped = [function(*inner_t) for inner_t in zip(*t)]
    else:
      mapped = function(*t)
    acc.append(mapped)
  return acc
