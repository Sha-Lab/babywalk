''' Batched Room-to-Room navigation environment '''

import numpy as np
import json
import networkx as nx
import os
import torch

from collections import namedtuple
from envs_utils import load_nav_graphs, structured_map
from model.cuda import try_cuda

WorldState = namedtuple("WorldState", ["scanId", "viewpointId", "viewIndex", "heading", "elevation"])
angle_inc = np.pi / 6.


class EnvBatch():
    ''' A simple wrapper for a batch of MatterSim environments,
        using discretized viewpoints and pretrained features '''
    
    def __init__(self, batch_size, adj_dict=None):
        self.batch_size = batch_size
        self.angle_inc = np.pi / 6.
        self.adj_dict = adj_dict
        if adj_dict is None:
            print("Error! Not load the adj dict!")
            exit(0)
    
    def _get_adj(self, state):
        query = '_'.join([state.scanId, state.viewpointId, str(state.viewIndex)])
        adj_list = self.adj_dict[query]
        return adj_list
    
    def getStartState(self, scanIds, viewpointIds, headings):
        def f(scanId, viewpointId, heading):
            elevation = 0
            viewIndex = (12 * round(elevation / self.angle_inc + 1) + round(heading / self.angle_inc) % 12)
            return WorldState(scanId=scanId,
                              viewpointId=viewpointId,
                              viewIndex=viewIndex,
                              heading=heading,
                              elevation=elevation)
        
        return structured_map(f, scanIds, viewpointIds, headings)
    
    def getAdjs(self, world_states):
        def f(world_state):
            return self._get_adj(world_state)
        
        return structured_map(f, world_states)
    
    def makeActions(self, world_states, actions, attrs):
        def f(world_state, action, loc_attrs):
            if action == 0:
                return world_state
            else:
                loc_attr = loc_attrs[action]
                return WorldState(scanId=world_state.scanId,
                                  viewpointId=loc_attr['nextViewpointId'],
                                  viewIndex=loc_attr['absViewIndex'],
                                  heading=(loc_attr['absViewIndex'] % 12) * self.angle_inc,
                                  elevation=(loc_attr['absViewIndex'] // 12 - 1) * self.angle_inc)
        
        return structured_map(f, world_states, actions, attrs)


class RoomEnv():
    ''' Implements the Room to Room navigation task, using discretized viewpoints and pretrained features '''
    
    @staticmethod
    def load_adj_feature(adj_list_file):
        with open(adj_list_file, 'r') as f:
            adj_dict = json.load(f)
        return adj_dict
    
    @staticmethod
    def load_nav_graphs():
        ''' Load connectivity graph for each scan, useful for reasoning about shortest paths '''
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
    def make_state_embeddings(state_embedding_size, states_map, image_features_list):
        state_embedding = {}
        for scan, state_list in states_map.items():
            embedding = np.zeros((len(state_list), 36, state_embedding_size))
            for i, state in enumerate(state_list):
                fake_state = {'scanId': scan,
                              'viewpointId': state}
                feature = [featurizer.get_features(fake_state) for featurizer in image_features_list][0]
                embedding[i] = feature
            state_embedding[scan] = try_cuda(torch.from_numpy(embedding).float())
        return state_embedding
    
    @staticmethod
    def build_viewpoint_loc_embedding(viewIndex):
        """
        Position embedding:
        heading 64D + elevation 64D
        1) heading: [sin(heading) for _ in range(1, 33)] +
                    [cos(heading) for _ in range(1, 33)]
        2) elevation: [sin(elevation) for _ in range(1, 33)] +
                      [cos(elevation) for _ in range(1, 33)]
        """
        embedding = np.zeros((36, 128), np.float32)
        for absViewIndex in range(36):
            relViewIndex = (absViewIndex - viewIndex) % 12 + (absViewIndex // 12) * 12
            rel_heading = (relViewIndex % 12) * angle_inc
            rel_elevation = (relViewIndex // 12 - 1) * angle_inc
            embedding[absViewIndex, 0:32] = np.sin(rel_heading)
            embedding[absViewIndex, 32:64] = np.cos(rel_heading)
            embedding[absViewIndex, 64:96] = np.sin(rel_elevation)
            embedding[absViewIndex, 96:] = np.cos(rel_elevation)
        return try_cuda(torch.from_numpy(embedding).float())
    
    def __init__(self, batch_size, paths, states_map, distances, state_embedding, loc_embeddings, adj_dict):
        self.env = EnvBatch(batch_size, adj_dict=adj_dict)
        self.margin = 3.0
        self.batch_size = batch_size
        self.paths = paths
        self.states_map = states_map
        self.distances = distances
        self.state_embedding = state_embedding
        self.loc_embeddings = loc_embeddings
    
    def _build_action_embedding(self, adj_loc_list, feature):
        feature_adj = feature[[adj_dict['absViewIndex'] for adj_dict in adj_loc_list]]
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
        return torch.cat((feature_adj, try_cuda(torch.from_numpy(embedding).float())), dim=-1)
    
    def _shortest_path_action(self, state, adj_loc_list, goalViewpointId):
        '''
        Determine next action on the shortest path to goal,
        for supervised training.
        '''
        if state.viewpointId == goalViewpointId:
            return 0
        for n_a, loc_attr in enumerate(adj_loc_list):
            if loc_attr['nextViewpointId'] == goalViewpointId:
                return n_a
        path = self.paths[state.scanId][state.viewpointId][goalViewpointId]
        nextViewpointId = path[1]
        for n_a, loc_attr in enumerate(adj_loc_list):
            if loc_attr['nextViewpointId'] == nextViewpointId:
                return n_a
        
        # Next nextViewpointId not found! This should not happen!
        print('adj_loc_list:', adj_loc_list)
        print('nextViewpointId:', nextViewpointId)
        long_id = '{}_{}'.format(state.scanId, state.viewpointId)
        print('longId:', long_id)
        raise Exception('Bug: nextViewpointId not in adj_loc_list')
    
    def _observe(self, world_states, include_teacher=True, include_instruction=True,
                 include_feature=True, step=None):
        obs = []
        for i, adj_loc_list in enumerate(self.env.getAdjs(world_states)):
            assert len(self.batch) == self.batch_size
            item = self.batch[i]
            state = self.world_states[i]
            ob = {
                'scan': state.scanId,
                'viewpoint': state.viewpointId,
                'viewIndex': state.viewIndex,
                'heading': state.heading,
                'elevation': state.elevation,
                'adj_loc_list': adj_loc_list,
                'instr_id': item['instr_id']
            }
            if include_instruction:
                ob['instr_encoding'] = item['instr_encoding']
                ob['instructions'] = item['instructions']
            if include_feature:
                idx = self.states_map[state.scanId].index(state.viewpointId)
                feature = self.state_embedding[state.scanId][idx]
                feature_with_loc = torch.cat((feature, self.loc_embeddings[state.viewIndex]), dim=-1)
                action_embedding = self._build_action_embedding(adj_loc_list, feature)
                ob['feature'] = [feature_with_loc]
                ob['action_embedding'] = action_embedding
            if include_teacher:
                ob['goal'] = item['path'][-1]
                if step is not None and (step + 1) < len(item['path']):
                    ob['teacher'] = item['path'][step + 1]
                else:
                    ob['teacher'] = item['path'][-1]
                ob['teacher_action'] = self._shortest_path_action(state, adj_loc_list, ob['teacher'])
            obs.append(ob)
        return obs
    
    def get_starting_obs(self, instance_list, step=None):
        scanIds = [item['scan'] for item in instance_list]
        viewpointIds = [item['path'][0] for item in instance_list]
        headings = [item['heading'] for item in instance_list]
        self.world_states = self.env.getStartState(scanIds, viewpointIds, headings)
        obs = self._observe(self.world_states, step=step)
        return obs
    
    def reset(self, next_batch=None, step=None):
        ''' Load a new minibatch / episodes. '''
        if next_batch:
            self.batch = next_batch
        obs = self.get_starting_obs(self.batch, step=step)
        return obs
    
    def step(self, obs, actions=None, step=None, include_instruction=True, reward_type='dis'):
        '''
        1. Take action (same interface as makeActions)
        2. Make obs
        3. Compute reward
        4. Done or not
        '''
        if actions is None:
            actions = [ob['teacher_action'] for ob in obs]
        dist2goal = [self.distances[ob['scan']][ob['viewpoint']][ob['goal']] for ob in obs]
        attrs = [ob['adj_loc_list'] for ob in obs]
        self.world_states = self.env.makeActions(self.world_states, actions, attrs)
        obs = self._observe(self.world_states, step=step, include_instruction=include_instruction)
        dist2goal_new = [self.distances[ob['scan']][ob['viewpoint']][ob['goal']] for ob in obs]
        shrink = 10
        if reward_type == 'cls' or reward_type == 'dtw':
            reward = [0 for _ in range(self.batch_size)]
        else:
            reward = [(dist2goal[i] - dist2goal_new[i]) / shrink if action != 0 else int(dist2goal[i] < self.margin)
                      for i, action in enumerate(actions)]
        done = (np.array(actions) == 0).astype(np.uint8)
        # obs, reward, done, info
        return obs, np.array(reward), done
    
    def shortest_paths_to_goals(self, obs, max_steps):
        all_obs = [[ob] for ob in obs]
        all_actions = [[] for _ in obs]
        ended = np.zeros(self.batch_size)
        for t in range(max_steps):
            actions = [ob['teacher_action'] for ob in obs]
            for i, a in enumerate(actions):
                if not ended[i]:
                    all_actions[i].append(a)
            obs, _, ended = self.step(obs, actions=actions, step=t + 1, include_instruction=False)
            for i, ob in enumerate(obs):
                if not ended[i] and t < max_steps - 1:
                    all_obs[i].append(ob)
            if ended.all():
                break
        return all_obs, all_actions
    
    def gold_obs_actions_and_instructions(self, batch, max_steps=100):
        obs = self.reset(next_batch=batch, step=0)
        path_obs, path_actions = self.shortest_paths_to_goals(obs, max_steps)
        encoded_instructions = [item['instr_encoding'] for item in batch]
        return path_obs, path_actions, encoded_instructions
    
    def get_dtw(self, scan, reference, prediction):
        margin = 3.0
        success = (self.distances[scan][prediction[-1]][reference[-1]] < margin)
        dtw_matrix = np.inf * np.ones((len(prediction) + 1, len(reference) + 1))
        dtw_matrix[0][0] = 0
        for i in range(1, len(prediction) + 1):
            for j in range(1, len(reference) + 1):
                best_previous_cost = min(
                    dtw_matrix[i - 1][j], dtw_matrix[i][j - 1], dtw_matrix[i - 1][j - 1])
                cost = self.distances[scan][prediction[i - 1]][reference[j - 1]]
                dtw_matrix[i][j] = cost + best_previous_cost
        dtw = dtw_matrix[len(prediction)][len(reference)]
        ndtw = np.exp(-dtw / (margin * len(reference)))
        return ndtw + success
    
    def get_cls(self, scan, reference, prediction):
        decay = 3
        success = (self.distances[scan][reference[-1]][prediction[-1]] < 3.0)
        pc = 0
        path_pl = 0
        traj_pl = 0
        for i, loc in enumerate(reference):
            if i < len(reference) - 1:
                path_pl += self.distances[scan][reference[i]][reference[i + 1]]
            nearest = np.inf
            for pred_loc in prediction:
                if self.distances[scan][loc][pred_loc] < nearest:
                    nearest = self.distances[scan][loc][pred_loc]
            pc += np.exp(-nearest / decay)
        pc /= len(reference)
        epl = pc * path_pl
        for i in range(len(prediction) - 1):
            traj_pl += self.distances[scan][prediction[i]][prediction[i + 1]]
        if epl == 0 and traj_pl == 0:
            cls = 0
        else:
            cls = pc * epl / (epl + abs(epl - traj_pl))
        return cls + success
