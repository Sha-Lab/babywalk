''' Utils for training and evaluation '''

import os
import json
import time
import math
import random
import torch
import numpy as np

from model.cuda import try_cuda
from src.process_data import make_data
from src.vocab.tokenizer import VOCAB_PAD_IDX, VOCAB_EOS_IDX
from simulator.envs.image_feature import ImageFeatures
from simulator.envs.env import RoomEnv
from simulator.envs.paths import ADJ_LIST_FILE


def random_seed():
  torch.manual_seed(1)
  torch.cuda.manual_seed(1)


def batch_observations_and_actions(path_obs, path_actions, padding_feature,
                                   padding_action):
  batch_size = len(path_obs)
  seq_lengths = np.array([len(a) for a in path_actions])
  max_path_length = seq_lengths.max()
  mask = np.ones((batch_size, max_path_length), np.uint8)
  image_features = [[] for _ in range(batch_size)]
  action_embeddings = [[] for _ in range(batch_size)]
  for i in range(batch_size):
    assert len(path_obs[i]) == len(path_actions[i])
    mask[i, :len(path_actions[i])] = 0
    image_features[i] = [ob['feature'][0] for ob in path_obs[i]]
    action_embeddings[i] = [ob['action_embedding'][path_actions[i][j]]
                            for j, ob in enumerate(path_obs[i])]
    image_features[i].extend([padding_feature]
                             * (max_path_length - len(path_actions[i])))
    action_embeddings[i].extend([padding_action]
                                * (max_path_length - len(path_actions[i])))
    image_features[i] = torch.stack(image_features[i], dim=0)
    action_embeddings[i] = torch.stack(action_embeddings[i], dim=0)
  batched_image_features = torch.stack(image_features, dim=0)
  batched_action_embeddings = torch.stack(action_embeddings, dim=0)
  mask = try_cuda(torch.from_numpy(mask).byte())
  return batched_image_features, batched_action_embeddings, mask, seq_lengths


def batch_instructions_from_encoded(encoded_instructions, max_length,
                                    reverse=False, cut=True):
  num_instructions = len(encoded_instructions)
  seq_tensor = np.full((num_instructions, max_length), VOCAB_PAD_IDX)
  seq_lengths = []
  for i, inst in enumerate(encoded_instructions):
    if len(inst) > 0 and inst[-1] == VOCAB_EOS_IDX:
      inst = inst[:-1]
    if reverse:
      inst = inst[::-1]
    inst = np.concatenate((inst, [VOCAB_EOS_IDX]))
    inst = inst[:max_length]
    seq_tensor[i, :len(inst)] = inst
    seq_lengths.append(len(inst))
  
  if cut:
    seq_tensor = torch.from_numpy(seq_tensor)[:, :max(seq_lengths)]
    mask = (seq_tensor == VOCAB_PAD_IDX)[:, :max(seq_lengths)]
  else:
    seq_tensor = torch.from_numpy(seq_tensor)
    mask = (seq_tensor == VOCAB_PAD_IDX)
  
  return try_cuda(seq_tensor.long()), try_cuda(mask.byte()), seq_lengths


def make_data_and_env(args):
  # make data
  train_data, val_data, all_val_data, vocab, tok, train_tag = make_data(args)
  
  # make env
  random_seed()
  image_features_list = ImageFeatures.from_args(args)
  paths, states_map, distances = RoomEnv.load_graphs()
  state_embedding = RoomEnv.make_state_embeddings(args, states_map,
                                                  image_features_list)
  loc_embeddings = [RoomEnv.build_viewpoint_loc_embedding(args, viewIndex)
                    for viewIndex in range(args.num_views)]
  adj_dict = RoomEnv.load_adj_feature(ADJ_LIST_FILE)
  env = RoomEnv(args, paths, states_map, distances, state_embedding,
                loc_embeddings, adj_dict)
  return train_data, val_data, all_val_data, env, vocab, tok, train_tag


def make_batch(data, ix, n_iter, batch_size, shuffle=True,
               sort_instr_len=True):
  batches = []
  new_ix = ix
  for i in range(n_iter):
    batch = data[new_ix:new_ix + batch_size]
    if len(batch) < batch_size:
      random.shuffle(data) if shuffle else None
      new_ix = batch_size - len(batch)
      batch += data[:new_ix]
    else:
      new_ix += batch_size
    if sort_instr_len:
      batch = sorted(batch, key=lambda item: item['instr_length'],
                     reverse=True)
    batches.append(batch)
  return batches, new_ix


def get_model_prefix(model_name, feedback_method):
  model_prefix = '{}_{}'.format(model_name, feedback_method)
  return model_prefix


def pretty_json_dump(obj, fp):
  json.dump(obj, fp, sort_keys=True, indent=4, separators=(',', ':'))


def as_minutes(s):
  m = math.floor(s / 60)
  s -= m * 60
  return '%dm %ds' % (m, s)


def time_since(since, percent):
  now = time.time()
  s = now - since
  es = s / (percent)
  rs = es - s
  return '%s (- %s)' % (as_minutes(s), as_minutes(rs))


def run(arg_parser, entry_function):
  arg_parser.add_argument("--pdb", action='store_true')
  arg_parser.add_argument("--ipdb", action='store_true')
  arg_parser.add_argument("--no_cuda", action='store_true')
  
  args = arg_parser.parse_args()
  
  import torch.cuda
  torch.cuda.disabled = args.no_cuda
  
  if args.ipdb:
    import ipdb
    ipdb.runcall(entry_function, args)
  elif args.pdb:
    import pdb
    pdb.runcall(entry_function, args)
  else:
    entry_function(args)


def check_dir(dir_list):
  for dir in dir_list:
    if not os.path.exists(dir):
      os.makedirs(dir)
