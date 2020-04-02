''' Utils for io, language, connectivity graphs etc '''

import os, sys

sys.path.append('.')

import json, time, math, random

from src.process_data import make_data
from simulator.envs.image_feature import ImageFeatures
from simulator.envs.env import RoomEnv
from simulator.envs.paths import adj_list_file
import torch


def random_seed():
    torch.manual_seed(1)
    torch.cuda.manual_seed(1)


def make_data_and_env(args):
    ### make data
    train_data, val_data, all_val_data, vocab, tok, train_tag = make_data(args)
    
    ### make env
    random_seed()
    image_features_list = ImageFeatures.from_args(args)
    paths, states_map, distances = RoomEnv.load_nav_graphs()
    state_embedding = RoomEnv.make_state_embeddings(ImageFeatures.feature_dim, states_map, image_features_list)
    loc_embeddings = [RoomEnv.build_viewpoint_loc_embedding(viewIndex) for viewIndex in range(36)]
    adj_dict = RoomEnv.load_adj_feature(adj_list_file)
    env = RoomEnv(args.batch_size, paths, states_map, distances, state_embedding, loc_embeddings, adj_dict)
    return train_data, val_data, all_val_data, env, vocab, tok, train_tag


def make_batch(data, ix, n_iter, batch_size, shuffle=True, sort_instr_len=True):
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
            batch = sorted(batch, key=lambda item: item['instr_length'], reverse=True)
        batches.append(batch)
    return batches, new_ix


def get_model_prefix(model_name, feedback_method):
    model_prefix = '{}_{}'.format(model_name, feedback_method)
    return model_prefix


def pretty_json_dump(obj, fp):
    json.dump(obj, fp, sort_keys=True, indent=4, separators=(',', ':'))


def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs))


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
