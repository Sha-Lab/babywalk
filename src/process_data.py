import math
import copy
import json
from collections import defaultdict
from src.vocab.vocab_path import TRAIN_VOCAB
from src.vocab.tokenizer import Tokenizer, read_vocab


def make_data(args):
  # determine splits
  if args.use_test:
    train_splits = []
    val_splits = ['test']
  elif args.task_name == 'R2T8':
    train_splits = []
    val_splits = ['R2R_val_unseen', 'R4R_val_unseen', 'R6R_val_unseen',
                  'R8R_val_unseen']
  elif args.task_name == 'R2R' or args.task_name == 'R4R':
    train_splits = ['train']
    val_splits = ['val_seen', 'val_unseen']
  else:
    train_splits = ['train']
    val_splits = ['val_unseen']
  
  if args.add_augment:
    train_splits.append(args.augment_data)
  vocab = read_vocab(TRAIN_VOCAB)
  tok = Tokenizer(vocab=vocab)
  
  # get datasets from file
  train_data = load_task_datasets(train_splits, args.task_name,
                                  postfix=args.split_postfix,
                                  tokenizer=tok,
                                  one_by_one_mode=args.one_by_one_mode)
  val_data = load_task_datasets(val_splits, args.task_name,
                                postfix=args.split_postfix,
                                tokenizer=tok,
                                one_by_one_mode=args.one_by_one_mode)
  
  # split for training
  if len(train_data) > 0:
    assert len(train_data['train']) >= args.batch_size, \
      "data not enough for one batch, reduce the batch size"
  if args.curriculum_rl:
    if args.one_by_one_mode == 'period':
      train_data = period_split_curriculum(train_data, tok, args.history)
    elif args.one_by_one_mode == 'landmark':
      train_data = landmark_split_curriculum(train_data, tok, args.history)
    else:
      raise ValueError("Error! One by one mode is not implemented.")
  elif args.il_mode is not None:
    if args.il_mode == 'period_split':
      train_data, counter = period_split(train_data, tok, 0, args.history)
      val_data, _ = period_split(val_data, tok, counter, args.history) \
        if not args.one_by_one else (val_data, None)
    elif args.il_mode == 'landmark_split':
      train_data, counter = landmark_split(train_data, tok, 0, args.history)
      val_data, _ = landmark_split(val_data, tok, counter, args.history) \
        if not args.one_by_one else (val_data, None)
    else:
      raise ValueError("Error! Training mode not available.")
  
  # make it together for evaluator
  train_tag = '-'.join(train_splits)
  train_data = merge_data(train_data)
  all_val_data = merge_data(val_data) if args.one_by_one_mode != 'landmark' \
    else merge_data_landmark(val_data)
  
  # make single data splitted sentence by sentence for evaluation
  if args.one_by_one:
    if args.one_by_one_mode == 'period':
      val_data = period_split_curriculum(val_data, tok, args.history,
                                         use_test=args.use_test)
    elif args.one_by_one_mode == 'landmark':
      val_data = landmark_split_curriculum(val_data, tok, args.history,
                                           use_test=args.use_test)
    else:
      print("Error! Not implemented one by one mode!")
      exit(0)
    val_data = {tag: sorted(data, key=lambda x: len(x))
                for tag, data in val_data.items()}
  return train_data, val_data, all_val_data, vocab, tok, train_tag


def merge_data(data):
  total_val = []
  for tag, d in data.items():
    total_val += d
  return total_val


def merge_data_landmark(data):
  total_val = []
  for tag, data in data.items():
    for d in data:
      new_d = dict(d)
      new_d['path'] = [d['path'][0][0]]
      for i in range(len(d['path'])):
        new_d['path'].extend(d['path'][i][1:])
      new_d['instructions'] = ' '.join(new_d['instructions'])
      total_val.append(new_d)
  return total_val


def load_dataset(split, task, postfix):
  data = []
  with open('tasks/%s/data/%s_%s%s.json' % (task, task, split, postfix)) as f:
    data += json.load(f)
    print("Load dataset %s_%s%s" % (task, split, postfix))
  return data


def load_task_datasets(splits, task, postfix='', tokenizer=None,
                       one_by_one_mode=None):
  dataset = {}
  id_list = defaultdict(lambda: 0)
  for split in splits:
    data = []
    for item in load_dataset(split, task, postfix):
      if one_by_one_mode == "landmark":
        new_item = dict(item)
        new_item['instr_id'] = '%s_%d' \
                               % (item['path_id'], id_list[item['path_id']])
        id_list[item['path_id']] += 1
        data.append(new_item)
      else:
        instructions = item['instructions']
        for j, instr in enumerate(instructions):
          new_item = dict(item)
          new_item['instr_id'] = '%s_%d' % (item['path_id'], j)
          new_item['instructions'] = instr
          if tokenizer:
            new_item['instr_encoding'], new_item[
              'instr_length'] = tokenizer.encode_sentence(instr)
          data.append(new_item)
    dataset[split] = data
  return dataset


def add_history_to_data(data, history_heading, history_path, history_instr,
                        history_instr_encoding):
  data['history_heading'] = copy.deepcopy(history_heading)
  data['history_path'] = copy.deepcopy(history_path)
  data['history_instr'] = copy.deepcopy(history_instr)
  data['history_instr_encoding'] = copy.deepcopy(history_instr_encoding)
  history_heading.append(data['heading'])
  history_path.append(data['path'])
  history_instr.append(data['instructions'])
  history_instr_encoding.append(data['instr_encoding'])


def period_split(datasets, tok, counter, history):
  splited = {}
  for tag, data in datasets.items():
    new_data = []
    for d in data:
      history_heading, history_path = [], []
      history_instr, history_instr_encoding = [], []
      ins = d['instructions']
      ins_splited = ins.split('.')
      if not ins_splited[0]:
        ins_splited = ins_splited[1:-1]
      else:
        ins_splited = ins_splited[:-1]
      ratio = 0
      last_path_split_point = 1
      ins_len = sum([len(ins_sp.split()) for ins_sp in ins_splited])
      for i, ins_sp in enumerate(ins_splited):
        ratio += len(ins_sp.split()) / ins_len
        if ratio > 1:
          ratio = 1
        path_split_point = math.ceil(len(d['path']) * ratio)
        
        new_d = copy.deepcopy(d)
        new_d['path_id'] = counter
        new_d['instr_id'] = str(counter) + '_0'
        new_d['path'] = d['path'][last_path_split_point - 1:path_split_point]
        new_d['instructions'] = ins_sp + '.'
        new_d['instr_encoding'], new_d['instr_length'] = tok.encode_sentence(
          ins_sp + '.')
        new_d['heading'] = d['headings'][last_path_split_point - 1]
        new_d['remain_split'] = len(ins_splited) - i
        if history:
          add_history_to_data(new_d, history_heading, history_path,
                              history_instr, history_instr_encoding)
        
        last_path_split_point = path_split_point
        new_data.append(new_d)
        counter += 1
      assert new_d['path'][-1] == d['path'][-1]
    splited[tag] = new_data
  return splited, counter


def landmark_split(datasets, tok, counter, history):
  splited = {}
  for tag, data in datasets.items():
    new_data = []
    for d in data:
      history_heading, history_path = [], []
      history_instr, history_instr_encoding = [], []
      for i, ins in enumerate(d['instructions']):
        new_d = copy.deepcopy(d)
        new_d['path_id'] = counter
        new_d['instr_id'] = str(counter) + '_0'
        new_d['path'] = d['path'][i]
        new_d['instructions'] = ins
        new_d['instr_encoding'], new_d['instr_length'] = tok.encode_sentence(
          ins)
        new_d['heading'] = float(d['headings'][i])
        new_d['remain_split'] = len(d['instructions']) - i
        if history:
          add_history_to_data(new_d, history_heading, history_path,
                              history_instr, history_instr_encoding)
        
        new_data.append(new_d)
        counter += 1
    splited[tag] = new_data
  return splited, counter


def period_split_curriculum(datasets, tok, history, use_test=False):
  splited = {}
  for tag, data in datasets.items():
    new_data = []
    for d in data:
      history_heading, history_path = [], []
      history_instr, history_instr_encoding = [], []
      new_d_list = []
      ins = d['instructions']
      ins_splited = ins.split('.')
      if not ins_splited[0]:
        ins_splited = ins_splited[1:-1]
      else:
        ins_splited = ins_splited[:-1]
      ratio = 0
      last_path_split_point = 1
      ins_len = sum([len(ins_sp.split()) for ins_sp in ins_splited])
      for i, ins_sp in enumerate(ins_splited):
        ratio += len(ins_sp.split()) / ins_len
        if ratio > 1:
          ratio = 1
        path_split_point = math.ceil(len(d['path']) * ratio)
        
        new_d = copy.deepcopy(d)
        new_d['instructions'] = ins_sp + '.'
        new_d['instr_encoding'], new_d['instr_length'] = \
          tok.encode_sentence(ins_sp + '.')
        if use_test:
          new_d['path'] = d['path'] * 2
          new_d['heading'] = float(d['heading'])
        else:
          new_d['path'] = d['path'][last_path_split_point - 1:path_split_point]
          new_d['heading'] = d['headings'][last_path_split_point - 1]
        if history:
          add_history_to_data(new_d, history_heading, history_path,
                              history_instr, history_instr_encoding)
        
        last_path_split_point = path_split_point
        new_d_list.append(new_d)
      assert new_d['path'][-1] == d['path'][-1]
      new_data.append(new_d_list)
    splited[tag] = new_data
  return splited


def landmark_split_curriculum(datasets, tok, history, use_test=False):
  splited = {}
  for tag, data in datasets.items():
    new_data = []
    for d in data:
      history_heading, history_path = [], []
      history_instr, history_instr_encoding = [], []
      new_d_list = []
      for i, ins in enumerate(d['instructions']):
        new_d = copy.deepcopy(d)
        new_d['instructions'] = ins
        new_d['instr_encoding'], new_d['instr_length'] = \
          tok.encode_sentence(ins)
        if use_test:
          new_d['path'] = d['path'] * 2
          new_d['heading'] = float(d['heading'])
        else:
          new_d['path'] = d['path'][i]
          new_d['heading'] = float(d['headings'][i])
        if history:
          add_history_to_data(new_d, history_heading, history_path,
                              history_instr, history_instr_encoding)
        
        new_d_list.append(new_d)
      new_data.append(new_d_list)
    splited[tag] = new_data
  return splited
