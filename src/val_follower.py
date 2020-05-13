import os
import os.path
import numpy as np
import argparse
import math
import sys

sys.path.append('.')

from src.params import add_general_args
from src.utils import check_dir, make_batch, get_model_prefix, run
from src.train_follower import train_setup
from simulator.envs.image_feature import ImageFeatures


def val(args, agent, val_data, evaluator):
  task_prefix = os.path.join('tasks', args.task_name)
  result_dir = os.path.join(task_prefix, args.result_dir)
  check_dir([result_dir])
  
  def make_path(dir):
    return os.path.join(dir, '%s_%s'
                        % (get_model_prefix(args.model_name, args.task_name),
                           'validation'))
  
  # run validation
  loss_str = ''
  ratio_number = []
  metric_dict = {
    'lengths': [], 'nav_error': [], 'sr': [], 'spl': [], 'cls': [], 'ndtw': [],
    'sdtw': [], 'sr_std': [], 'spl_std': [], 'cls_std': []
  }
  for tag, d in val_data.items():
    ratio_number.append(len(d))
    it = math.ceil(len(d) / args.batch_size)
    test_batch, _ = make_batch(d, 0, it, args.batch_size,
                               shuffle=False, sort_instr_len=False)
    agent.test(test_batch, one_by_one=args.one_by_one, history=args.history,
               exp_forget=args.exp_forget)
    agent.results_path = make_path(result_dir) + '_' + tag + '.json'
    
    print("evaluating on {}".format(tag))
    score_summary = evaluator.score_results(agent.results, update_results=True)
    loss_str += '\n%s' % (tag)
    for metric, val in sorted(score_summary.items()):
      if metric in metric_dict:
        metric_dict[metric].append(val)
      loss_str += ', %s: %.3f' % (metric, val)
    agent.write_results()
    print("PL: %.2f, NE: %.2f, SR: %.1f, SPL: %.1f, "
          "CLS: %.1f, NDTW: %.1f, SDTW %.1f"
          % (metric_dict['lengths'][-1], metric_dict['nav_error'][-1],
             metric_dict['sr'][-1] * 100, metric_dict['spl'][-1] * 100,
             metric_dict['cls'][-1] * 100, metric_dict['ndtw'][-1] * 100,
             metric_dict['sdtw'][-1] * 100))
  
  print("Average\nPL: %.2f, NE: %.2f, SR: %.1f, SPL: %.1f, "
        "CLS: %.1f, NDTW: %.1f, SDTW %.1f"
        % (np.array(metric_dict['lengths']).mean(),
           np.array(metric_dict['nav_error']).mean(),
           np.array(metric_dict['sr']).mean() * 100,
           np.array(metric_dict['spl']).mean() * 100,
           np.array(metric_dict['cls']).mean() * 100,
           np.array(metric_dict['ndtw']).mean() * 100,
           np.array(metric_dict['sdtw']).mean() * 100))
  print('%s' % (loss_str))


def test(args, agent, val_data):
  task_prefix = os.path.join('tasks', args.task_name)
  result_dir = os.path.join(task_prefix, args.result_dir)
  check_dir([result_dir])
  
  def make_path(dir):
    return os.path.join(dir, '%s_%s'
                        % (get_model_prefix(args.model_name, args.task_name),
                           'test'))
  
  # test
  for _, d in val_data.items():
    it = math.ceil(len(d) / args.batch_size)
    test_batch, _ = make_batch(d, 0, it, args.batch_size,
                               shuffle=False, sort_instr_len=False)
    agent.test(test_batch, one_by_one=args.one_by_one, history=args.history,
               exp_forget=args.exp_forget)
    agent.results_path = make_path(result_dir) + '.json'
    
    # reformat
    reformat_results = []
    for id, r in agent.results.items():
      reformat_results.append({
        "instr_id": id,
        "trajectory": [[r["trajectory"][i]] + list(r["trajectory_radians"][i])
                       for i in range(len(r["trajectory"]))]
      })
    agent.results = reformat_results
    agent.write_results()


def train_val(args):
  ''' Validate on seen and unseen splits. '''
  follower, _, val_data, evaluator, _, _ = train_setup(args)
  if args.use_test:
    test(args, follower, val_data)
  else:
    val(args, follower, val_data, evaluator)


def make_arg_parser():
  parser = argparse.ArgumentParser()
  ImageFeatures.add_args(parser)
  add_general_args(parser)
  return parser


if __name__ == "__main__":
  run(make_arg_parser(), train_val)
