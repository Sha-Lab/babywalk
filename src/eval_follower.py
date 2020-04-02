''' Evaluation of agent trajectories '''

from collections import defaultdict
import numpy as np
import copy
from collections import namedtuple

EvalResult = namedtuple("EvalResult", "nav_error, oracle_error, trajectory_steps, trajectory_length, "
                                      "stop, acc, ssr, sr, osr, spl, cls, ndtw, sdtw")


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
    
    def _get_nearest(self, scan, goal_id, path):
        near_id = path[0]
        near_d = self.distances[scan][near_id][goal_id]
        for item in path:
            d = self.distances[scan][item][goal_id]
            if d < near_d:
                near_id = item
                near_d = d
        return near_id
    
    def _score_item(self, instr_id, path):
        ''' Calculate error based on the final position in trajectory, and also
            the closest position (oracle stopping rule). '''
        gt = self.gt[int(instr_id.split('_')[0])]
        goal = gt['path'][-1]
        final_position = path[-1]
        nearest_position = self._get_nearest(gt['scan'], goal, path)
        distance = self.distances[gt['scan']][path[0]][goal]
        nav_error = self.distances[gt['scan']][final_position][goal]
        oracle_error = self.distances[gt['scan']][nearest_position][goal]
        trajectory_steps = len(path) - 1
        trajectory_length = 0
        prev = path[0]
        for curr in path[1:]:
            trajectory_length += self.distances[gt['scan']][prev][curr]
            prev = curr
        
        stop = (path[-1] == path[-2])
        acc = (nav_error == 0) and stop
        sr = nav_error < self.margin
        ssr = sr and stop
        osr = oracle_error < self.margin
        spl = sr * distance / max(trajectory_length, distance) if distance > 0 else sr
        cls = self.cls(gt, path)
        dtw = self.dtw(gt['scan'], path, gt['path'])
        ndtw = np.exp(-dtw / (self.margin * len(path)))
        sdtw = ndtw * sr
        return EvalResult(nav_error=nav_error, oracle_error=oracle_error,
                          trajectory_steps=trajectory_steps,
                          trajectory_length=trajectory_length,
                          stop=stop, acc=acc, ssr=ssr, sr=sr, osr=osr, spl=spl,
                          cls=cls, ndtw=ndtw, sdtw=sdtw)
    
    def dtw(self, scan, prediction, reference):
        dtw_matrix = np.inf * np.ones((len(prediction) + 1, len(reference) + 1))
        dtw_matrix[0][0] = 0
        for i in range(1, len(prediction) + 1):
            for j in range(1, len(reference) + 1):
                best_previous_cost = min(
                    dtw_matrix[i - 1][j], dtw_matrix[i][j - 1], dtw_matrix[i - 1][j - 1])
                cost = self.distances[scan][prediction[i - 1]][reference[j - 1]]
                dtw_matrix[i][j] = cost + best_previous_cost
        dtw = dtw_matrix[len(prediction)][len(reference)]
        return dtw
    
    def length(self, scan, nodes):
        return np.sum([self.distances[scan][edge[0]][edge[1]] for edge in zip(nodes[:-1], nodes[1:])])
    
    def cls(self, item, trajectory):
        decay = 3
        scan = item['scan']
        path = item['path']
        pc = np.mean([np.exp(-np.min([self.distances[scan][u][v] for v in trajectory]) / decay) for u in path])
        epl = pc * self.length(scan, path)
        traj_pl = self.length(scan, trajectory)
        if epl == 0 and traj_pl == 0:
            cls = 0
        else:
            cls = pc * epl / (epl + abs(epl - traj_pl))
        return cls
    
    def score_results(self, results, update_results=False):
        # results should be a dictionary mapping instr_ids to dictionaries,
        # with each dictionary containing (at least) a 'trajectory' field
        self.scores = defaultdict(list)
        model_scores = []
        instr_ids = set(self.instr_ids)
        
        instr_count = 0
        for instr_id, result in results.items():
            if instr_id in instr_ids:
                instr_count += 1
                instr_ids.remove(instr_id)
                
                eval_result = self._score_item(instr_id, result['trajectory'])
                self.scores['nav_errors'].append(eval_result.nav_error)
                self.scores['oracle_errors'].append(eval_result.oracle_error)
                self.scores['trajectory_steps'].append(
                    eval_result.trajectory_steps)
                self.scores['trajectory_lengths'].append(
                    eval_result.trajectory_length)
                self.scores['stop'].append(eval_result.stop)
                self.scores['acc'].append(eval_result.acc)
                self.scores['ssr'].append(eval_result.ssr)
                self.scores['sr'].append(eval_result.sr)
                self.scores['cls'].append(eval_result.cls)
                self.scores['osr'].append(eval_result.osr)
                self.scores['spl'].append(eval_result.spl)
                self.scores['ndtw'].append(eval_result.ndtw)
                self.scores['sdtw'].append(eval_result.sdtw)
                if update_results:
                    result['nav_errors'] = eval_result.nav_error
                    result['oracle_errors'] = eval_result.oracle_error
                    result['trajectory_steps'] = eval_result.trajectory_steps
                    result['trajectory_lengths'] = eval_result.trajectory_length
                    result['stop'] = eval_result.stop
                    result['acc'] = eval_result.acc
                    result['ssr'] = eval_result.ssr
                    result['sr'] = eval_result.sr
                    result['osr'] = eval_result.osr
                    result['spl'] = eval_result.spl
                    result['cls'] = eval_result.cls
                    result['ndtw'] = eval_result.ndtw
                    result['sdtw'] = eval_result.sdtw
                    result['expert_trajectory'] = self.gt[int(instr_id.split('_')[0])]['path']
                    result['distance'] = self.gt[int(instr_id.split('_')[0])]['distance']
                    result['scan'] = self.gt[int(instr_id.split('_')[0])]['scan']
                    result['instruction'] = self.gt[int(instr_id.split('_')[0])]['instructions'][
                        int(instr_id.split('_')[1])]
                
                if 'score' in result:
                    model_scores.append(result['score'])
        
        score_summary = {
            'nav_error': np.average(self.scores['nav_errors']),
            'oracle_error': np.average(self.scores['oracle_errors']),
            'steps': np.average(self.scores['trajectory_steps']),
            'lengths': np.average(self.scores['trajectory_lengths']),
            'cls': np.average(self.scores['cls']),
            'acc': float(sum(self.scores['acc']) / instr_count),
            'stop': float(sum(self.scores['stop']) / len(self.scores['stop'])),
            'sr': float(sum(self.scores['sr']) / len(self.scores['sr'])),
            'ssr': float(sum(self.scores['ssr'])) / len(self.scores['ssr']),
            'osr': float(sum(self.scores['osr']) / len(self.scores['osr'])),
            'spl': float(sum(self.scores['spl']) / len(self.scores['spl'])),
            'ndtw': float(sum(self.scores['ndtw']) / len(self.scores['ndtw'])),
            'sdtw': float(sum(self.scores['sdtw']) / len(self.scores['sdtw'])),
        }
        if update_results:
            score_summary['sr_std'] = np.std(self.scores['sr'])
            score_summary['cls_std'] = np.std(self.scores['cls'])
            score_summary['spl_std'] = np.std(self.scores['spl'])
        if len(model_scores) > 0:
            score_summary['model_score'] = np.average(model_scores)
        
        return score_summary, self.scores
