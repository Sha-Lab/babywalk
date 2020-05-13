import os
import time
import numpy as np
import pandas as pd
import argparse
import math
import sys

sys.path.append('.')

from torch import optim
from collections import defaultdict
from tensorboardX import SummaryWriter
from src.vocab.vocab_path import GLOVE_PATH
from src.vocab.tokenizer import VOCAB_PAD_IDX
from src.utils import time_since, check_dir, make_batch, get_model_prefix, \
  make_data_and_env, run
from src.params import add_general_args
from simulator.envs.image_feature import ImageFeatures
from model.speaker_lstm import SpeakerEncoderLSTM, SpeakerDecoderLSTM
from src.speaker import Seq2SeqSpeaker
from model.follower_coattend import EncoderLSTM, AttnDecoderLSTM
from model.follower_coground import CogroundDecoderLSTM
from model.cuda import try_cuda
from src.follower import Seq2SeqFollower
from src.eval_follower import FollowerEvaluation


def train(args, agent, train_data, val_data, evaluator, speaker, train_tag):
  def make_path(dir, n_iter):
    return os.path.join(dir, '%s_%s_iter_%d' % (
      get_model_prefix(args.model_name, args.feedback_method),
      train_tag, n_iter))
  
  # check result directories
  task_prefix = os.path.join('tasks', args.task_name)
  result_dir = os.path.join(task_prefix, args.result_dir)
  snapshot_dir = os.path.join(task_prefix, args.snapshot_dir)
  plot_dir = os.path.join(task_prefix, args.plot_dir)
  summary_dir = os.path.join(task_prefix, args.summary_dir,
                             time.strftime("%Y-%m-%d-%H-%M-%S", time.gmtime()))
  check_dir([result_dir, snapshot_dir, plot_dir, summary_dir])
  
  # initialize
  data_log = defaultdict(list)
  n_iters = args.n_iters
  log_every = args.log_every
  best_metrics = {}
  last_model_saved = {}
  writer = SummaryWriter(log_dir=summary_dir)
  train_ix = 0
  
  print('Training with %s feedback' % args.feedback_method)
  start = time.time()
  for idx in range(0, n_iters, log_every):
    interval = min(log_every, n_iters - idx)
    iter = idx + interval
    data_log['iteration'].append(iter)
    
    # make mini-batch
    if args.curriculum_rl:
      train_batch, train_ix = make_batch(train_data, train_ix, interval,
                                         args.batch_size, sort_instr_len=False)
    else:
      train_batch, train_ix = make_batch(train_data, train_ix, interval,
                                         args.batch_size, sort_instr_len=True)
    
    # train
    if args.curriculum_rl:
      assert args.reward == True
      agent.train_crl(train_batch, interval, speaker,
                      curriculum=args.count_curriculum, history=args.history,
                      reward_type=args.reward_type, exp_forget=args.exp_forget,
                      beam_size=args.beam_size, feedback=args.feedback_method)
    elif args.reward:
      agent.train_reward(train_batch, interval, speaker,
                         history=args.history, reward_type=args.reward_type,
                         exp_forget=args.exp_forget, beam_size=args.beam_size,
                         feedback=args.feedback_method)
    else:
      agent.train(train_batch, interval,
                  history=args.history, exp_forget=args.exp_forget,
                  feedback=args.feedback_method)
    
    # output loss / reward
    train_loss_avg = np.array(agent.losses).mean()
    data_log['train loss'].append(train_loss_avg)
    loss_str = 'train loss: %.4f' % train_loss_avg
    writer.add_scalar('data/train_loss', train_loss_avg, iter)
    
    if args.reward:
      int_reward_avg = np.array(agent.int_rewards).mean()
      data_log['int_reward'].append(int_reward_avg)
      loss_str += ', internal reward: %.4f' % int_reward_avg
      writer.add_scalar('data/int_reward', int_reward_avg, iter)
      
      ext_reward_avg = np.array(agent.ext_rewards).mean()
      data_log['ext_reward'].append(ext_reward_avg)
      loss_str += ', external reward: %.4f' % ext_reward_avg
      writer.add_scalar('data/ext_reward', ext_reward_avg, iter)
    
    # run validation
    save_log = []
    for tag, d in val_data.items():
      it = math.ceil(len(d) / args.batch_size)
      test_batch, _ = make_batch(d, 0, it, args.batch_size,
                                 shuffle=False, sort_instr_len=False)
      agent.test(test_batch, history=args.history, one_by_one=args.one_by_one,
                 exp_forget=args.exp_forget)
      agent.results_path = make_path(result_dir, iter) + '_' + tag + '.json'
      
      # evaluate results
      print("evaluating on {}".format(tag))
      score_summary = evaluator.score_results(agent.results)
      
      loss_str += '\n%s' % (tag)
      for metric, val in sorted(score_summary.items()):
        data_log['%s %s' % (tag, metric)].append(val)
        writer.add_scalar('data/' + tag + '_' + metric, val, iter)
        if metric in ['sr', 'cls', 'sdtw']:
          print("%s on %s: %.3f" % (metric, tag, val))
          
          # save model
          key = (tag, metric)
          if key not in best_metrics or best_metrics[key] < val:
            best_metrics[key] = val
            if not args.no_save:
              model_path = make_path(snapshot_dir, iter) \
                           + "_%s-%s=%.3f" % (tag, metric, val)
              save_log.append("new best, saved model to %s" % model_path)
              agent.save(model_path)
              if key in last_model_saved:
                for old_model_path in \
                  agent.encoder_and_decoder_paths(last_model_saved[key]):
                  os.remove(old_model_path)
              last_model_saved[key] = model_path
        loss_str += ', %s: %.3f' % (metric, val)
    
    # report training process
    print(('%s (%d %d%%) %s'
           % (time_since(start, float(iter) / n_iters), iter,
              float(iter) / n_iters * 100, loss_str)))
    for s in save_log:
      print(s)
    if not args.no_save:
      if args.save_every and iter % args.save_every == 0:
        agent.save(make_path(snapshot_dir, iter))
      df = pd.DataFrame(data_log)
      df.set_index('iteration')
      df_path = '%s%s_%s_log.csv' \
                % (plot_dir,
                   get_model_prefix(args.model_name, args.feedback_method),
                   train_tag)
      df.to_csv(df_path)
    
    # update curriculum
    if args.curriculum_rl \
      and iter % args.curriculum_iters == 0 \
      and args.count_curriculum < args.max_curriculum:
      args.count_curriculum += 1
      agent.reset_rav()
      agent.encoder_optimizer, agent.decoder_optimizer = \
        reset_optimizer(args, agent.encoder, agent.decoder)


def reset_optimizer(args, encoder, decoder):
  def filter_param(param_list):
    return [p for p in param_list if p.requires_grad]
  
  enc_para = encoder.parameters()
  dec_para = decoder.parameters()
  if args.learning_method == "adam":
    encoder_optimizer = optim.Adam(filter_param(enc_para), lr=args.lr,
                                   weight_decay=args.weight_decay)
    decoder_optimizer = optim.Adam(filter_param(dec_para), lr=args.lr,
                                   weight_decay=args.weight_decay)
  elif args.learning_method == "sgd":
    encoder_optimizer = optim.SGD(filter_param(enc_para), lr=args.lr,
                                  momentum=0.9, nesterov=True,
                                  weight_decay=args.weight_decay)
    decoder_optimizer = optim.SGD(filter_param(dec_para), lr=args.lr,
                                  momentum=0.9, nesterov=True,
                                  weight_decay=args.weight_decay)
  elif args.learning_method == "rms":
    encoder_optimizer = optim.RMSprop(filter_param(enc_para), lr=args.lr,
                                      weight_decay=args.weight_decay)
    decoder_optimizer = optim.RMSprop(filter_param(dec_para), lr=args.lr,
                                      weight_decay=args.weight_decay)
  else:
    raise ValueError("Error! Learning method not correct")
  
  return encoder_optimizer, decoder_optimizer


def make_speaker_models(args, vocab_size, env, tok):
  glove = np.load(GLOVE_PATH)
  encoder = SpeakerEncoderLSTM(args.feature_size, args.hidden_size,
                               args.dropout)
  decoder = SpeakerDecoderLSTM(vocab_size, args.wemb, VOCAB_PAD_IDX,
                               args.hidden_size, args.dropout, glove=glove)
  encoder = try_cuda(encoder)
  decoder = try_cuda(decoder)
  
  agent = Seq2SeqSpeaker(env, "", args, encoder, decoder, tok)
  return agent


def make_follower_models(args, vocab_size, all_val_data, env):
  glove = np.load(GLOVE_PATH)
  encoder = EncoderLSTM(vocab_size, args.wemb, args.hidden_size, VOCAB_PAD_IDX,
                        args.dropout, glove=glove)
  if args.coground:
    decoder = CogroundDecoderLSTM(args.action_embed_size, args.hidden_size,
                                  args.dropout, args.feature_size,
                                  args.max_ins_len, history=args.history)
  else:
    decoder = AttnDecoderLSTM(args.action_embed_size, args.hidden_size,
                              args.dropout, args.feature_size,
                              history=args.history,
                              lstm_mem=args.exp_forget < 0)
  
  encoder = try_cuda(encoder)
  decoder = try_cuda(decoder)
  encoder_optimizer, decoder_optimizer = \
    reset_optimizer(args, encoder, decoder)
  
  agent = Seq2SeqFollower(env, "", args, encoder, decoder, encoder_optimizer,
                          decoder_optimizer)
  evaluator = FollowerEvaluation(env, all_val_data)
  return agent, evaluator


def train_setup(args):
  ''' Load data, setup environment and setup agent '''
  train_data, val_data, all_val_data, env, vocab, tok, train_tag = \
    make_data_and_env(args)
  agent, evaluator = make_follower_models(args, len(vocab), all_val_data, env)
  if args.reward and not args.no_speaker:
    speaker = make_speaker_models(args, len(vocab), env, tok)
    speaker.load(args.speaker_prefix, **{})
    print("Load speaker model %s" % args.speaker_prefix)
  else:
    speaker = None
  if args.follower_prefix is not None:
    agent.load(args.follower_prefix, args.load_opt, **{})
    print("Load follower model %s" % args.follower_prefix)
  return agent, train_data, val_data, evaluator, speaker, train_tag


def train_val(args):
  ''' Train on the training set, and validate on validation (seen/unseen) set. '''
  agent, train_data, val_data, evaluator, speaker, train_tag = \
    train_setup(args)
  train(args, agent, train_data, val_data, evaluator, speaker, train_tag)


def make_arg_parser():
  parser = argparse.ArgumentParser()
  ImageFeatures.add_args(parser)
  add_general_args(parser)
  return parser


if __name__ == "__main__":
  run(make_arg_parser(), train_val)
