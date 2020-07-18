''' Batched Room-to-Room navigation environment '''

import numpy as np
import json
import networkx as nx
import os
import torch

from collections import namedtuple
from envs_utils import load_nav_graphs, structured_map
from model.cuda import try_cuda

ANGLE_INC = np.pi / 6.
WorldState = namedtuple(
  "WorldState",
  ["scan_id", "viewpoint_id", "view_index", "heading", "elevation"]
)


class EnvBatch():
  ''' A simple wrapper for a batch of MatterSim environments,
      using discretized viewpoints and pretrained features,
      using an adjacency dictionary to replace the MatterSim simulator
  '''
  
  def __init__(self, adj_dict=None):
    self.adj_dict = adj_dict
    assert adj_dict is not None, "Error! No adjacency dictionary!"
  
  def get_start_state(self, scan_ids, viewpoint_ids, headings):
    def f(scan_id, viewpoint_id, heading):
      elevation = 0
      view_index = (12 * round(elevation / ANGLE_INC + 1)
                    + round(heading / ANGLE_INC) % 12)
      return WorldState(scan_id=scan_id,
                        viewpoint_id=viewpoint_id,
                        view_index=view_index,
                        heading=heading,
                        elevation=elevation)
    
    return structured_map(f, scan_ids, viewpoint_ids, headings)
  
  def get_adjs(self, world_states):
    def f(world_state):
      query = '_'.join([world_state.scan_id,
                        world_state.viewpoint_id,
                        str(world_state.view_index)])
      return self.adj_dict[query]
    
    return structured_map(f, world_states)
  
  def make_actions(self, world_states, actions, attrs):
    def f(world_state, action, loc_attrs):
      if action == 0:
        return world_state
      else:
        loc_attr = loc_attrs[action]
        return WorldState(scan_id=world_state.scan_id,
                          viewpoint_id=loc_attr['nextViewpointId'],
                          view_index=loc_attr['absViewIndex'],
                          heading=(loc_attr['absViewIndex'] % 12) * ANGLE_INC,
                          elevation=(loc_attr['absViewIndex'] // 12 - 1)
                                    * ANGLE_INC)
    
    return structured_map(f, world_states, actions, attrs)


class RoomEnv():
  ''' Implements the R2R (R4R, R6R, etc.) navigation task,
      using discretized viewpoints and pretrained features.
  '''
  
  @staticmethod
  def load_adj_feature(adj_list_file):
    with open(adj_list_file, 'r') as f:
      adj_dict = json.load(f)
    return adj_dict
  
  @staticmethod
  def load_graphs():
    ''' Load connectivity graph for each scan. '''
    scans = []
    for file in os.listdir("simulator/connectivity"):
      if file.endswith(".json"):
        scans.append(file.split('_')[0])
    print('Loading navigation graphs for %d scans' % len(scans))
    graphs = load_nav_graphs(scans)
    paths = {}
    matrix = {}
    states_map = {}
    distances = {}
    for scan, G in graphs.items():  # compute all shortest paths
      paths[scan] = dict(nx.all_pairs_dijkstra_path(G))
      matrix[scan] = nx.to_numpy_matrix(G)
      states_map[scan] = list(G.nodes)
      distances[scan] = dict(nx.all_pairs_dijkstra_path_length(G))
    return paths, states_map, distances
  
  @staticmethod
  def make_state_embeddings(args, states_map, image_features_list):
    state_embedding = {}
    for scan, state_list in states_map.items():
      embedding = np.zeros((len(state_list), args.num_views,
                            args.mean_pooled_dim))
      for i, state in enumerate(state_list):
        fake_state = {'scan_id': scan,
                      'viewpoint_id': state}
        feature = [featurizer.get_features(fake_state)
                   for featurizer in image_features_list][0]
        embedding[i] = feature
      state_embedding[scan] = torch.from_numpy(embedding).float()
    return state_embedding
  
  @staticmethod
  def build_viewpoint_loc_embedding(args, view_index):
    """
    Position embedding: heading 64D + elevation 64D
    1) heading: [sin(heading) for _ in range(1, 33)] +
                [cos(heading) for _ in range(1, 33)]
    2) elevation: [sin(elevation) for _ in range(1, 33)] +
                  [cos(elevation) for _ in range(1, 33)]
    """
    embedding = np.zeros((args.num_views, 128), np.float32)
    for abs_view_index in range(args.num_views):
      rel_view_index = (abs_view_index - view_index) % 12 \
                       + (abs_view_index // 12) * 12
      rel_heading = (rel_view_index % 12) * ANGLE_INC
      rel_elevation = (rel_view_index // 12 - 1) * ANGLE_INC
      embedding[abs_view_index, 0:32] = np.sin(rel_heading)
      embedding[abs_view_index, 32:64] = np.cos(rel_heading)
      embedding[abs_view_index, 64:96] = np.sin(rel_elevation)
      embedding[abs_view_index, 96:] = np.cos(rel_elevation)
    return torch.from_numpy(embedding).float()
  
  def __init__(self, args, paths, states_map, distances, state_embedding,
               loc_embeddings, adj_dict):
    self.env = EnvBatch(adj_dict=adj_dict)
    self.margin = 3.0
    self.paths = paths
    self.states_map = states_map
    self.distances = distances
    self.state_embedding = state_embedding
    self.loc_embeddings = loc_embeddings
    self.padding_action = try_cuda(torch.zeros(args.action_embed_size))
    self.padding_feature = try_cuda(torch.zeros(args.num_views,
                                                args.action_embed_size))
    self.shrink = 10  # shrink distance 10 times
  
  def _build_action_embedding(self, adj_loc_list, feature):
    feature_adj = feature[[adj_dict['absViewIndex']
                           for adj_dict in adj_loc_list]]
    feature_adj[0] = 0
    embedding = np.zeros((len(adj_loc_list), 128), np.float32)
    for a, adj_dict in enumerate(adj_loc_list):
      if a == 0:
        continue
      else:
        rel_heading = adj_dict['rel_heading']
        rel_elevation = adj_dict['rel_elevation']
      embedding[a][0:32] = np.sin(rel_heading)
      embedding[a][32:64] = np.cos(rel_heading)
      embedding[a][64:96] = np.sin(rel_elevation)
      embedding[a][96:] = np.cos(rel_elevation)
    angle_embed = torch.from_numpy(embedding).float()
    return try_cuda(torch.cat((feature_adj, angle_embed), dim=-1))
  
  def _build_feature_embedding(self, view_index, feature):
    angle_embed = self.loc_embeddings[view_index]
    return try_cuda(torch.cat((feature, angle_embed), dim=-1))
  
  def _shortest_path_action(self, state, adj_loc_list, goal_id):
    ''' Determine next action on the shortest path to goal, for supervised training. '''
    if state.viewpoint_id == goal_id:
      return 0
    for n_a, loc_attr in enumerate(adj_loc_list):
      if loc_attr['nextViewpointId'] == goal_id:
        return n_a
    path = self.paths[state.scan_id][state.viewpoint_id][goal_id]
    next_viewpoint_id = path[1]
    for n_a, loc_attr in enumerate(adj_loc_list):
      if loc_attr['nextViewpointId'] == next_viewpoint_id:
        return n_a
    
    # Next viewpoint_id not found! This should not happen!
    print('adj_loc_list:', adj_loc_list)
    print('next_viewpoint_id:', next_viewpoint_id)
    print('longId:', '{}_{}'.format(state.scan_id, state.viewpoint_id))
    raise Exception('Error: next_viewpoint_id not in adj_loc_list')
  
  def _observe(self, world_states, include_feature=True,
               include_teacher=True, step=None):
    """
    Return the observations of a batch of states
    :param world_states: states defined as a namedtuple
    :param done: has done, no need to provide ob
    :param include_feature: whether or not to return the pretrained features
    :param include_teacher: whether or not to return a teacher viewpoint and
                            teacher action (for supervision)
    :param step: step number in the gold trajectory
    :return: a list of observations, each is a dictionary
    """
    obs = []
    for i, adj_loc_list in enumerate(self.env.get_adjs(world_states)):
      item = self.batch[i]
      state = self.world_states[i]
      ob = {
        'scan': state.scan_id,
        'viewpoint': state.viewpoint_id,
        'view_index': state.view_index,
        'heading': state.heading,
        'elevation': state.elevation,
        'adj_loc_list': adj_loc_list,
        'instr_id': item['instr_id']
      }
      if include_feature:
        idx = self.states_map[state.scan_id].index(state.viewpoint_id)
        feature = self.state_embedding[state.scan_id][idx]
        feature_with_loc = self._build_feature_embedding(state.view_index,
                                                         feature)
        action_embedding = self._build_action_embedding(adj_loc_list, feature)
        ob['feature'] = [feature_with_loc]
        ob['action_embedding'] = action_embedding
      if include_teacher:
        ob['goal'] = item['path'][-1]
        if step is not None and (step + 1) < len(item['path']):
          ob['teacher'] = item['path'][step + 1]
        else:
          ob['teacher'] = item['path'][-1]
        ob['teacher_action'] = self._shortest_path_action(state, adj_loc_list,
                                                          ob['teacher'])
      obs.append(ob)
    return obs
  
  def reset(self, next_batch, step=None):
    ''' Load a new mini-batch and return the initial observation'''
    self.batch = next_batch
    scan_ids = [item['scan'] for item in next_batch]
    viewpoint_ids = [item['path'][0] for item in next_batch]
    headings = [item['heading'] for item in next_batch]
    self.world_states = self.env.get_start_state(scan_ids, viewpoint_ids,
                                                 headings)
    obs = self._observe(self.world_states, step=step)
    return obs
  
  def step(self, obs, actions, step=None):
    ''' Take one step from the current state
    :param obs: last observations
    :param actions: current actions
    :param step: step information for teacher action supervision
    :return: current observations, and "done" (finish or not)
    '''
    attrs = [ob['adj_loc_list'] for ob in obs]
    self.world_states = self.env.make_actions(self.world_states, actions,
                                              attrs)
    obs = self._observe(self.world_states, step=step)
    done = (np.array(actions) == 0).astype(np.uint8)
    return obs, done
  
  def _paths_to_goals(self, obs, max_steps):
    all_obs = [[ob] for ob in obs]
    all_actions = [[] for _ in obs]
    ended = np.zeros(len(obs))
    for t in range(max_steps):
      actions = [ob['teacher_action'] for ob in obs]
      for i, a in enumerate(actions):
        if not ended[i]:
          all_actions[i].append(a)
      obs, ended = self.step(obs, actions, step=t + 1)
      for i, ob in enumerate(obs):
        if not ended[i] and t < max_steps - 1:
          all_obs[i].append(ob)
      if ended.all():
        break
    return all_obs, all_actions
  
  def gold_obs_actions_and_instructions(self, batch, max_steps=100):
    obs = self.reset(batch, step=0)
    path_obs, path_actions = self._paths_to_goals(obs, max_steps)
    encoded_instructions = [item['instr_encoding'] for item in batch]
    return path_obs, path_actions, encoded_instructions
  
  def length(self, scan, nodes):
    return float(np.sum([self.distances[scan][edge[0]][edge[1]]
                         for edge in zip(nodes[:-1], nodes[1:])]))
  
  def get_mix(self, scan, prediction, reference):
    success = self.distances[scan][prediction[-1]][reference[-1]] < self.margin
    pad = [0] * (len(prediction) - 1)
    final = self.ndtw(scan, prediction, reference) * success \
            + self.cls(scan, prediction, reference)
    return pad + [final]
  
  def get_ndtw(self, scan, prediction, reference):
    success = self.distances[scan][prediction[-1]][reference[-1]] < self.margin
    pad = [0] * (len(prediction) - 2)
    return pad + [self.ndtw(scan, prediction, reference) + success]
  
  def ndtw(self, scan, prediction, reference):
    dtw_matrix = np.inf * np.ones((len(prediction) + 1, len(reference) + 1))
    dtw_matrix[0][0] = 0
    for i in range(1, len(prediction) + 1):
      for j in range(1, len(reference) + 1):
        best_previous_cost = min(dtw_matrix[i - 1][j],
                                 dtw_matrix[i][j - 1],
                                 dtw_matrix[i - 1][j - 1])
        cost = self.distances[scan][prediction[i - 1]][reference[j - 1]]
        dtw_matrix[i][j] = cost + best_previous_cost
    dtw = dtw_matrix[len(prediction)][len(reference)]
    ndtw = np.exp(-dtw / (self.margin * len(reference)))
    return ndtw
  
  def get_cls(self, scan, prediction, reference):
    success = self.distances[scan][reference[-1]][prediction[-1]] < self.margin
    pad = [0] * (len(prediction) - 2)
    return pad + [self.cls(scan, prediction, reference) + success]
  
  def cls(self, scan, prediction, reference):
    coverage = np.mean([np.exp(
      -np.min([self.distances[scan][u][v] for v in prediction]) / self.margin
    ) for u in reference])
    expected = coverage * self.length(scan, reference)
    score = expected \
            / (expected + np.abs(expected - self.length(scan, prediction)))
    return coverage * score
  
  def get_dis(self, scan, prediction, reference):
    goal = reference[-1]
    success = self.distances[scan][goal][prediction[-1]] < self.margin
    dis = [(self.distances[scan][goal][prediction[i]]
            - self.distances[scan][goal][prediction[i + 1]]) / self.shrink
           for i in range(len(prediction) - 1)]
    return dis[:-1] + [success]
