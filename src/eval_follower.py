''' Evaluation of agent trajectories '''

from collections import defaultdict
import numpy as np
import copy
from collections import namedtuple

EvalResult = namedtuple("EvalResult", "nav_error, oracle_error, "
                                      "trajectory_steps, trajectory_length, "
                                      "sr, osr, spl, cls, ndtw, sdtw")


class FollowerEvaluation(object):
  ''' Results submission format:
      [{'instr_id': string,
        'trajectory':[viewpoint_id]}] '''
  
  def __init__(self, env, data):
    self.margin = 3.0
    self.gt = {}
    self.instr_ids = []
    for item in data:
      if item['path_id'] not in self.gt:
        self.gt[item['path_id']] = copy.deepcopy(item)
        self.gt[item['path_id']]['instructions'] = [item['instructions']]
      else:
        self.gt[item['path_id']]['instructions'].append(item['instructions'])
      self.instr_ids.append(item['instr_id'])
    self.instr_ids = set(self.instr_ids)
    self.distances = env.distances
    self.env = env
  
  def _get_nearest(self, scan, goal_id, path):
    near_id = path[0]
    near_d = self.distances[scan][near_id][goal_id]
    for item in path:
      d = self.distances[scan][item][goal_id]
      if d < near_d:
        near_id = item
        near_d = d
    return near_id
  
  def _score_item(self, gt, path):
    ''' Calculate error based on the final position in trajectory, and also
        the closest position (oracle stopping rule). '''
    goal = gt['path'][-1]
    final_position = path[-1]
    nearest_position = self._get_nearest(gt['scan'], goal, path)
    dis = self.distances[gt['scan']][path[0]][goal]
    nav_error = self.distances[gt['scan']][final_position][goal]
    oracle_error = self.distances[gt['scan']][nearest_position][goal]
    trajectory_steps = len(path) - 1
    trajectory_length = self.env.length(gt['scan'], path)
    sr = nav_error < self.margin
    osr = oracle_error < self.margin
    spl = sr * dis / max(trajectory_length, dis) if dis > 0 else sr
    cls = self.env.cls(gt['scan'], path, gt['path'])
    ndtw = self.env.ndtw(gt['scan'], path, gt['path'])
    sdtw = ndtw * sr
    return EvalResult(nav_error=nav_error, oracle_error=oracle_error,
                      trajectory_steps=trajectory_steps,
                      trajectory_length=trajectory_length,
                      sr=sr, osr=osr, spl=spl, cls=cls, ndtw=ndtw, sdtw=sdtw)
  
  def score_results(self, results, update_results=False):
    '''
    evaluation on different metrics
    :param results: results should be a dictionary mapping instr_ids to
                    dictionaries, with each dictionary containing (at least)
                    a 'trajectory' field
    :param update_results: update the result dictionary for saving result files
    :return:
    '''
    self.scores = defaultdict(list)
    model_scores = []
    instr_ids = set(self.instr_ids)
    
    instr_count = 0
    for instr_id, result in results.items():
      if instr_id in instr_ids:
        instr_count += 1
        instr_ids.remove(instr_id)
        
        gt = self.gt[int(instr_id.split('_')[0])]
        eval_result = self._score_item(gt, result['trajectory'])
        self.scores['nav_errors'].append(eval_result.nav_error)
        self.scores['oracle_errors'].append(eval_result.oracle_error)
        self.scores['trajectory_steps'].append(eval_result.trajectory_steps)
        self.scores['trajectory_lengths'].append(eval_result.trajectory_length)
        self.scores['sr'].append(eval_result.sr)
        self.scores['cls'].append(eval_result.cls)
        self.scores['osr'].append(eval_result.osr)
        self.scores['spl'].append(eval_result.spl)
        self.scores['ndtw'].append(eval_result.ndtw)
        self.scores['sdtw'].append(eval_result.sdtw)
        if 'score' in result:
          model_scores.append(result['score'])
        if update_results:
          result['nav_errors'] = eval_result.nav_error
          result['oracle_errors'] = eval_result.oracle_error
          result['trajectory_steps'] = eval_result.trajectory_steps
          result['trajectory_lengths'] = eval_result.trajectory_length
          result['sr'] = eval_result.sr
          result['osr'] = eval_result.osr
          result['spl'] = eval_result.spl
          result['cls'] = eval_result.cls
          result['ndtw'] = eval_result.ndtw
          result['sdtw'] = eval_result.sdtw
          result['expert_trajectory'] = gt['path']
          result['distance'] = gt['distance']
          result['scan'] = gt['scan']
          result['instruction'] = \
            gt['instructions'][int(instr_id.split('_')[1])]
    
    score_summary = {
      'nav_error': np.average(self.scores['nav_errors']),
      'oracle_error': np.average(self.scores['oracle_errors']),
      'steps': np.average(self.scores['trajectory_steps']),
      'lengths': np.average(self.scores['trajectory_lengths']),
      'cls': np.average(self.scores['cls']),
      'sr': float(sum(self.scores['sr']) / len(self.scores['sr'])),
      'osr': float(sum(self.scores['osr']) / len(self.scores['osr'])),
      'spl': float(sum(self.scores['spl']) / len(self.scores['spl'])),
      'ndtw': float(sum(self.scores['ndtw']) / len(self.scores['ndtw'])),
      'sdtw': float(sum(self.scores['sdtw']) / len(self.scores['sdtw'])),
    }
    if len(model_scores) > 0:
      score_summary['model_score'] = np.average(model_scores)
    if update_results:
      score_summary['sr_std'] = np.std(self.scores['sr'])
      score_summary['cls_std'] = np.std(self.scores['cls'])
      score_summary['spl_std'] = np.std(self.scores['spl'])
    
    return score_summary
