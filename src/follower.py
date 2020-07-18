import numpy as np
import math
import copy
import torch
import torch.nn.functional as F
import torch.distributions as D

from torch.nn.utils.rnn import pad_sequence
from src.vocab.tokenizer import VOCAB_EOS_IDX
from src.utils import pretty_json_dump, batch_instructions_from_encoded, \
  batch_observations_and_actions
from model.cuda import try_cuda


class Seq2SeqFollower(object):
  ''' An agent based on an LSTM seq2seq model with attention. '''
  
  def __init__(self, env, results_path, args, encoder, decoder,
               encoder_optimizer, decoder_optimizer,
               reverse_instruction=True, max_sub_tasks=6):
    self.env = env
    self.results_path = results_path
    self.hidden_size = args.hidden_size
    self.encoder = encoder
    self.decoder = decoder
    self.encoder_optimizer = encoder_optimizer
    self.decoder_optimizer = decoder_optimizer
    self.episode_len = args.max_steps
    self.results = {}
    self.losses = []
    self.reverse_instruction = reverse_instruction
    self.max_instruction_length = args.max_ins_len
    self.rav, self.rav_count = 0, 0
    self.rav_step, self.rav_count_step = \
      [0] * self.episode_len * max_sub_tasks, \
      [0] * self.episode_len * max_sub_tasks
    self.max_sub_tasks = max_sub_tasks
  
  def reset_rav(self):
    self.rav, self.rav_count = 0, 0
    self.rav_step, self.rav_count_step = \
      [0] * self.episode_len * self.max_sub_tasks, \
      [0] * self.episode_len * self.max_sub_tasks
  
  def write_results(self):
    with open(self.results_path, 'w') as f:
      pretty_json_dump(self.results, f)
  
  def _rollout(self, batch, feedback, reward_flag=False, history=False,
               exp_forget=0.5):
    batch_size = len(batch)
    
    # Embed history memory
    if history:
      history_context = self._make_history_context(batch, decay=exp_forget)
    else:
      history_context = None
    
    # Batch instructions
    seq, seq_mask, seq_lengths = \
      batch_instructions_from_encoded([b['instr_encoding'] for b in batch],
                                      self.max_instruction_length,
                                      reverse=self.reverse_instruction)
    
    # Reset environment
    done = np.zeros(batch_size, dtype=np.uint8)
    obs = self.env.reset(batch)
    
    # Record starting point
    loss = 0
    count_valid = 0
    traj = [{'instr_id': ob['instr_id'],
             'scores': [],
             'heading': ob['heading'],
             'trajectory': [ob['viewpoint']],
             'trajectory_radians': [(ob['heading'], ob['elevation'])],
             'reward': [], } for ob in obs]
    
    # Init text embed and action
    ctx, h_t, c_t = self.encoder(seq, seq_lengths)
    u_t_prev = self.env.padding_action.expand(batch_size, -1)
    
    # Do a sequence rollout and calculate the loss
    sequence_scores = try_cuda(torch.zeros(batch_size))
    for t in range(self.episode_len):
      f_t = self._feature_variables(obs)
      all_u_t, action_mask = self._action_variable(obs)
      h_t, c_t, alpha, logit, alpha_v = \
        self.decoder(u_t_prev, all_u_t, f_t, h_t, c_t, ctx,
                     ctx_mask=seq_mask, history_context=history_context)
      
      # Supervised training
      target = self._teacher_action(obs, done)
      
      logit[action_mask] = -float('inf')
      if torch.isnan(logit).sum():
        raise ValueError("Error! network produce nan result!")
      
      # Determine next model inputs
      if feedback == 'teacher':
        a_t = torch.clamp(target, min=0)
      elif feedback == 'argmax':
        _, a_t = logit.max(1)
        a_t = a_t.detach()
      elif feedback == 'sample':
        probs = F.softmax(logit, dim=1)
        m = D.Categorical(probs)
        a_t = m.sample()
      else:
        raise ValueError("Error! Invalid feedback option!")
      
      # Valid count (not done yet trajectories)
      count_valid += len(obs) - done.sum()
      
      # Update the previous action
      u_t_prev = all_u_t[np.arange(batch_size), a_t, :].detach()
      action_scores = -F.cross_entropy(logit, a_t.clone(),
                                       ignore_index=-1, reduction='none')
      action_scores[done] = 0
      sequence_scores += action_scores
      
      # Calculate loss
      loss += self._criterion(logit, target)
      
      # Make environment action
      a_t[done] = 0
      obs, next_done = self.env.step(obs, a_t.tolist())
      
      # Save trajectory output
      for i, ob in enumerate(obs):
        if not done[i]:
          if reward_flag:
            traj[i]['scores'].append(-action_scores[i])
          traj[i]['trajectory'].append(ob['viewpoint'])
          traj[i]['trajectory_radians'].append((ob['heading'],
                                                ob['elevation']))
      
      # Early exit if all ended
      done = next_done
      if done.all():
        break
    
    for i, ob in enumerate(obs):
      traj[i]['score'] = sequence_scores[i].item() / len(traj[i]['trajectory'])
    return traj, loss / count_valid
  
  def _make_history_context(self, batch, decay=0.5):
    ''' Embed history context both vision and text, return a list of [vision embed, text embed]'''
    history_lengths = [len(b['history_heading']) for b in batch]
    max_history = max(history_lengths)
    context_list = []
    text_context_list = []
    for hist_count in range(max_history):
      new_batch = [copy.deepcopy(b) for b in batch]
      zero_list = []
      for i, b in enumerate(new_batch):
        if len(b['history_heading']) > hist_count:
          b['heading'] = b['history_heading'][hist_count]
          b['path'] = b['history_path'][hist_count]
          b['instr_encoding'] = b['history_instr_encoding'][hist_count]
        else:
          b['path'] = [b['path'][0]]
          b['instr_encoding'] = np.array([VOCAB_EOS_IDX])
          zero_list.append(i)
      path_obs, path_actions, encoded_instructions = \
        self.env.gold_obs_actions_and_instructions(new_batch)
      batched_image_features, batched_action_embeddings, _, seq_lengths = \
        batch_observations_and_actions(path_obs, path_actions,
                                       self.env.padding_feature,
                                       self.env.padding_action)
      seq_lengths[zero_list] = 0
      context = self.decoder(batched_image_features, batched_action_embeddings,
                             seq_lengths, context=True)
      context_list.append(context)
      max_len = max([len(ins) for ins in encoded_instructions])
      batched_ins, _, ins_lengths \
        = batch_instructions_from_encoded(encoded_instructions, max_len + 2,
                                          cut=False)
      text_context = self.encoder(batched_ins, ins_lengths, context=True)
      text_context_list.append(text_context)
    
    context_list = torch.stack(context_list, dim=1) if context_list else []
    text_context_list = \
      torch.stack(text_context_list, dim=1) if text_context_list else []
    if decay < 0:  # smaller than 0, use LSTM memory
      context = self.decoder.context_lstm(context_list, history_lengths)
      text_context = self.decoder.text_context_lstm(text_context_list,
                                                    history_lengths)
    else:  # not smaller than 0, use exp forget
      if len(context_list) > 0:
        exp_weight = np.zeros((len(history_lengths), max_history))
        for i, h in enumerate(history_lengths):
          exp_weight[i][:h] = [np.exp(-x * decay) for x in range(h)][::-1]
        exp_weight = F.normalize(try_cuda(
          torch.from_numpy(exp_weight)).float(), p=1, dim=1).unsqueeze(-1)
        context = (context_list * exp_weight).sum(dim=1)
        text_context = (text_context_list * exp_weight).sum(dim=1)
      else:
        context = try_cuda(torch.zeros(len(history_lengths), self.hidden_size))
        text_context = try_cuda(torch.zeros(len(history_lengths),
                                            self.hidden_size))
    return [context, text_context]
  
  def _feature_variables(self, obs):
    ''' Extract precomputed features into variable. '''
    feature_lists = list(zip(*[ob['feature'] for ob in obs]))
    return torch.stack(feature_lists[0])
  
  def _action_variable(self, obs):
    ''' Get the available action embedding for the agent to select.'''
    max_num_a = max([len(ob['adj_loc_list']) for ob in obs])
    is_valid = np.zeros((len(obs), max_num_a), np.float32)
    action_embeddings = []
    for i, ob in enumerate(obs):
      is_valid[i, len(ob['adj_loc_list']):] = 1
      action_embeddings.append(ob['action_embedding'])
    # action embed and action mask
    return pad_sequence(action_embeddings, batch_first=True), \
           try_cuda(torch.from_numpy(is_valid).byte())
  
  def _teacher_action(self, obs, ended):
    ''' Extract teacher actions into variable. '''
    a = torch.LongTensor(len(obs))
    for i, ob in enumerate(obs):
      a[i] = ob['teacher_action'] if not ended[i] else -1
    return try_cuda(a)
  
  def _criterion(self, logit, target):
    return F.cross_entropy(logit, target, ignore_index=-1, reduction='sum')
  
  def _get_reward(self, scan, prediction, reference, reward_type):
    ''' Compute the reward for DIS, CLS and DTW'''
    if reward_type == 'cls':
      return self.env.get_cls(scan, prediction, reference)
    elif reward_type == 'dtw':
      return self.env.get_ndtw(scan, prediction, reference)
    elif reward_type == 'dis':
      return self.env.get_dis(scan, prediction, reference)
    elif reward_type == 'mix':
      return self.env.get_mix(scan, prediction, reference)
    else:
      raise ValueError("Error! No such reward type!")
  
  def _get_decay_reward(self, beam_size, int_rewards, results, gamma=0.95):
    adv_array = np.zeros((beam_size, len(int_rewards[0]),
                          self.max_sub_tasks * self.episode_len))
    for beam in range(beam_size):
      R_intr = int_rewards[beam]
      for idx, f_result in enumerate(results[beam]):
        a_len = len(f_result['reward'])
        R_extr = 0
        for j in range(a_len):
          R_extr = R_extr * gamma + f_result['reward'][a_len - 1 - j]
          Adv = (R_extr + R_intr[idx])
          adv_array[beam, idx, j] = Adv
          self.rav = (self.rav * self.rav_count + Adv) \
                             / (self.rav_count + 1)
          self.rav_count += 1
    return adv_array
  
  def _get_loss_from_reward(self, beam_size, adv_array, results):
    loss = 0
    count_valid = 0
    for beam in range(beam_size):
      for idx, f_result in enumerate(results[beam]):
        a_len = len(f_result['scores'])
        count_valid += a_len
        for j in range(a_len):
          loss += (adv_array[beam, idx, j] - self.rav) \
                  * f_result['scores'][a_len - 1 - j]
    return loss / count_valid
  
  def train_reward(self, batch, n_iters, speaker, feedback='sample',
                   beam_size=8, history=False, reward_type='dis',
                   exp_forget=0.5):
    ''' Train for a given number of iterations with RL'''
    self.encoder.train()
    self.decoder.train()
    self.losses = []
    self.int_rewards = []
    self.ext_rewards = []
    delta = 0.5
    for i in range(n_iters):
      follower_results_cache = []
      int_reward_cache = []
      ext_reward_cache = []
      self.encoder_optimizer.zero_grad()
      self.decoder_optimizer.zero_grad()
      
      # Rollout and compute reward
      for beam in range(beam_size):
        follower_results, _ = \
          self._rollout(batch[i], feedback, reward_flag=True, history=history,
                        exp_forget=exp_forget)
        for j, result in enumerate(follower_results):
          result['reward'] = self._get_reward(batch[i][j]['scan'],
                                              result['trajectory'],
                                              batch[i][j]['path'], reward_type)
        ext_reward_cache.append([f_result['reward'][-1]
                                 for f_result in follower_results])
        follower_results_cache.append(follower_results)
        if speaker is not None:
          speaker_results = speaker.query(batch[i], follower_results,
                                          feedback='teacher')
          int_reward_cache.append([s_result['score'] * delta
                                   for s_result in speaker_results])
        else:
          int_reward_cache.append([0] * len(batch[i]))
      
      # Get decayed reward for each action
      adv_array = self._get_decay_reward(beam_size, int_reward_cache,
                                         follower_results_cache)
      
      # Compute loss and backward
      loss = self._get_loss_from_reward(beam_size, adv_array,
                                        follower_results_cache)
      loss.backward()
      torch.nn.utils.clip_grad_norm_(self.encoder.parameters(), 1.)
      torch.nn.utils.clip_grad_norm_(self.decoder.parameters(), 1.)
      self.encoder_optimizer.step()
      self.decoder_optimizer.step()
      
      # Record rewards and loss
      self.int_rewards.append(int_reward_cache)
      self.ext_rewards.append(ext_reward_cache)
      self.losses.append(loss.item())
  
  def _curriculum_rollout(self, batch, feedback, curriculum, reward_type,
                          history=True, exp_forget=0.5):
    # curriculum is total steps
    max_cur = min(curriculum, max([len(item) for item in batch]))
    start_idx = [len(item) - max_cur if len(item) >= max_cur else 0
                 for item in batch]
    start_point = [item[start_idx[j]]['path'][0]
                   for j, item in enumerate(batch)]
    start_heading = [item[start_idx[j]]['heading']
                     for j, item in enumerate(batch)]
    rollout_results = [{'instr_id': item[0]['instr_id'],
                        'trajectory': [start_point[j]],
                        'scores': []} for j, item in enumerate(batch)]
    if history:
      history_heading = [copy.deepcopy(item[start_idx[j]]['history_heading'])
                         for j, item in enumerate(batch)]
      history_path = [copy.deepcopy(item[start_idx[j]]['history_path'])
                      for j, item in enumerate(batch)]
    for i in range(max_cur):
      # make batch for rollout
      current_batch = []
      not_done_idxs = []
      for j, b in enumerate(batch):
        if start_idx[j] + i < len(b):
          new_item = copy.deepcopy(b[start_idx[j] + i])
          new_item['path'] = [start_point[j],
                              b[-1]['path'][-1]]  # use final goal as objective
          new_item['heading'] = start_heading[j]
          if history:
            new_item['history_heading'] = history_heading[j]
            new_item['history_path'] = history_path[j]
          current_batch.append(new_item)
          not_done_idxs.append(j)
      follower_results, _ = self._rollout(current_batch, feedback,
                                          reward_flag=True, history=history,
                                          exp_forget=exp_forget)
      
      # record rollout results
      for j, f_result in enumerate(follower_results):
        k = not_done_idxs[j]  # the original indices in rollout
        start_point[k] = f_result['trajectory'][-1]
        start_heading[k] = f_result['trajectory_radians'][-1][0]
        rollout_results[k]['trajectory'].extend(f_result['trajectory'][1:])
        rollout_results[k]['scores'].extend(f_result['scores'])
        if history:
          history_heading[k].append(f_result['heading'])
          history_path[k].append(f_result['trajectory'])
          if rollout_results[k]['trajectory'][-1] == \
            rollout_results[k]['trajectory'][-2]:
            history_path[k][-1] = history_path[k][-1][:-1]
        if rollout_results[k]['trajectory'][-1] == \
          rollout_results[k]['trajectory'][-2]:
          rollout_results[k]['trajectory'] = \
            rollout_results[k]['trajectory'][:-1]
    
    # Compute full trajectory and the reward of actions
    for j, r_result in enumerate(rollout_results):
      fixed_path = []
      total_path = []
      for i in range(start_idx[j]):
        fixed_path.extend(batch[j][i]['path'][:-1])
      for i in range(len(batch[j])):
        total_path.extend(batch[j][i]['path'][:-1])
      total_path.append(batch[j][-1]['path'][-1])
      r_result['trajectory'] = fixed_path + r_result['trajectory']
      full_reward = self._get_reward(batch[j][-1]['scan'],
                                     rollout_results[j]['trajectory'],
                                     total_path, reward_type)
      r_result['reward'] = full_reward[-len(r_result['scores']):]
    return rollout_results
  
  def train_crl(self, batch, n_iters, speaker, feedback='sample', beam_size=8,
                history=True, curriculum=0, reward_type='dis', exp_forget=0.5):
    ''' Train for a given number of iterations with CRL'''
    self.encoder.train()
    self.decoder.train()
    self.losses = []
    self.int_rewards = []
    self.ext_rewards = []
    delta = 0.5
    small_batch_size = int(len(batch[0]) / (curriculum + 1))
    accumulate_steps = math.ceil(len(batch[0]) / small_batch_size)
    for i in range(n_iters):
      self.encoder_optimizer.zero_grad()
      self.decoder_optimizer.zero_grad()
      avg_int_reward = 0
      avg_ext_reward = 0
      avg_loss = 0
      
      # Accumulate gradient
      for step in range(accumulate_steps):
        follower_results_cache = []
        int_reward_cache = []
        ext_reward_cache = []
        small_batch = batch[i][step * small_batch_size:
                               (step + 1) * small_batch_size]
        batch_ratio = len(small_batch) / len(batch[i])
        
        # Rollout and compute reward
        for beam in range(beam_size):
          follower_results = \
            self._curriculum_rollout(small_batch, feedback, curriculum + 1,
                                     reward_type, history=history,
                                     exp_forget=exp_forget)
          follower_results_cache.append(follower_results)
          ext_reward_cache.append([f_result['reward'][-1]
                                   for f_result in follower_results])
          if speaker is not None:
            speaker_results = speaker.query(small_batch, follower_results,
                                            feedback='teacher',
                                            curriculum=True)
            int_reward_cache.append([s_result['score'] * delta
                                     for s_result in speaker_results])
          else:
            int_reward_cache.append([0] * len(small_batch))
        
        # Get decayed reward for each action
        adv_array = self._get_decay_reward(beam_size, int_reward_cache,
                                           follower_results_cache)
        
        # Compute loss and backward
        loss = self._get_loss_from_reward(beam_size, adv_array,
                                          follower_results_cache)
        loss = loss * batch_ratio
        loss.backward()

        avg_int_reward += np.array(int_reward_cache).mean() * batch_ratio
        avg_ext_reward += np.array(ext_reward_cache).mean() * batch_ratio
        avg_loss += loss.item()

      torch.nn.utils.clip_grad_norm_(self.encoder.parameters(), 1.)
      torch.nn.utils.clip_grad_norm_(self.decoder.parameters(), 1.)
      self.encoder_optimizer.step()
      self.decoder_optimizer.step()
      
      # Record rewards and loss
      self.int_rewards.append(avg_int_reward)
      self.ext_rewards.append(avg_ext_reward)
      self.losses.append(avg_loss)
  
  def train(self, batch, n_iters, feedback='sample', history=False,
            exp_forget=0.5):
    ''' Train for a given number of iterations with IL'''
    self.encoder.train()
    self.decoder.train()
    self.losses = []
    for i in range(n_iters):
      self.encoder_optimizer.zero_grad()
      self.decoder_optimizer.zero_grad()
      _, loss = self._rollout(batch[i], feedback,
                              history=history, exp_forget=exp_forget)
      loss.backward()
      self.losses.append(loss.item())
      torch.nn.utils.clip_grad_norm_(self.encoder.parameters(), 10.)
      torch.nn.utils.clip_grad_norm_(self.decoder.parameters(), 10.)
      self.encoder_optimizer.step()
      self.decoder_optimizer.step()
  
  def test(self, batch, feedback='argmax', one_by_one=False, history=False,
           exp_forget=0.5):
    ''' Evaluate once on each instruction in the current environment '''
    with torch.no_grad():
      self.encoder.eval()
      self.decoder.eval()
      
      self.results = {}
      count = 0
      looped = False
      while True:
        if one_by_one:
          max_executes = max([len(b) for b in batch[count]])
          rollout_results = [
            {'instr_id': item[0]['instr_id'],
             'trajectory': [item[0]['path'][0]],
             'trajectory_radians': [(item[0]['heading'], 0.0)],
             'segments': len(item)}
            for item in batch[count]]
          start_point = [item[0]['path'][0] for item in batch[count]]
          start_heading = [item[0]['heading'] for item in batch[count]]
          history_heading = [[] for _ in range(len(batch[count]))]
          history_path = [[] for _ in range(len(batch[count]))]
          for i in range(max_executes):
            current_batch = []
            not_done_idxs = []
            for j, b in enumerate(batch[count]):
              if i < len(b):
                new_item = copy.deepcopy(b[i])
                new_item['path'] = [start_point[j]]
                new_item['heading'] = start_heading[j]
                if history:
                  new_item['history_heading'] = history_heading[j]
                  new_item['history_path'] = history_path[j]
                current_batch.append(new_item)
                not_done_idxs.append(j)
            follower_results, _ = self._rollout(current_batch, feedback,
                                                history=history,
                                                exp_forget=exp_forget)
            for j, f_result in enumerate(follower_results):
              k = not_done_idxs[j]  # the original indices in rollout
              start_point[k] = f_result['trajectory'][-1]
              start_heading[k] = f_result['trajectory_radians'][-1][0]
              rollout_results[k]['trajectory'].extend(
                f_result['trajectory'][1:])
              rollout_results[k]['trajectory_radians'].extend(
                f_result['trajectory_radians'][1:])
              if history:
                history_heading[k].append(f_result['heading'])
                history_path[k].append(f_result['trajectory'])
                if rollout_results[k]['trajectory'][-1] == \
                  rollout_results[k]['trajectory'][-2]:
                  history_path[k][-1] = history_path[k][-1][:-1]
              if rollout_results[k]['trajectory'][-1] == \
                rollout_results[k]['trajectory'][-2]:
                rollout_results[k]['trajectory'] = \
                  rollout_results[k]['trajectory'][:-1]
                rollout_results[k]['trajectory_radians'] = \
                  rollout_results[k]['trajectory_radians'][:-1]
        else:
          rollout_results, _ = self._rollout(batch[count], feedback,
                                             history=history,
                                             exp_forget=exp_forget)
        
        count += 1
        for result in rollout_results:
          if result['instr_id'] in self.results:
            looped = True
          else:
            self.results[result['instr_id']] = result
        if looped or len(batch) == count:
          break
    return self.results
  
  def encoder_and_decoder_paths(self, base_path):
    return base_path + "_enc", base_path + "_dec", \
           base_path + "_enc_opt", base_path + "_dec_opt"
  
  def save(self, path):
    ''' Snapshot models '''
    encoder_path, decoder_path, encoder_opt_path, decoder_opt_path = \
      self.encoder_and_decoder_paths(path)
    torch.save(self.encoder.state_dict(), encoder_path)
    torch.save(self.decoder.state_dict(), decoder_path)
    torch.save(self.encoder_optimizer.state_dict(), encoder_opt_path)
    torch.save(self.decoder_optimizer.state_dict(), decoder_opt_path)
  
  def load(self, path, load_opt, **kwargs):
    ''' Loads parameters (but not training state) '''
    encoder_path, decoder_path, encoder_opt_path, decoder_opt_path = \
      self.encoder_and_decoder_paths(path)
    self.encoder.load_my_state_dict(torch.load(encoder_path, **kwargs))
    self.decoder.load_my_state_dict(torch.load(decoder_path, **kwargs))
    if load_opt:
      self.encoder_optimizer.load_state_dict(
        torch.load(encoder_opt_path, **kwargs))
      self.decoder_optimizer.load_state_dict(
        torch.load(decoder_opt_path, **kwargs))
