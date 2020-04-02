''' Agents: stop/random/shortest/seq2seq  '''

import sys
import numpy as np
import copy
import torch
from torch.autograd import Variable
import torch.nn.functional as F
import torch.distributions as D
from torch.nn.utils.rnn import pad_sequence

from src.vocab.tokenizer import vocab_pad_idx, vocab_eos_idx
from src.utils import pretty_json_dump
from model.cuda import try_cuda

NUM_VIEWS = 36


class Seq2SeqFollower(object):
    ''' An agent based on an LSTM seq2seq model with attention. '''
    
    def __init__(self, env, results_path, args, encoder, decoder, encoder_optimizer, decoder_optimizer,
                 reverse_instruction=True, max_sub_tasks=6):
        self.env = env
        self.results_path = results_path
        self.batch_size = args.batch_size
        self.hidden_size = args.hidden_size
        self.encoder = encoder
        self.decoder = decoder
        self.encoder_optimizer = encoder_optimizer
        self.decoder_optimizer = decoder_optimizer
        self.episode_len = args.max_steps
        self.results = {}
        self.losses = []
        self.acces = []
        self.reverse_instruction = reverse_instruction
        self.max_instruction_length = args.max_ins_len
        self.padding_action = try_cuda(torch.zeros(args.action_embed_size))
        self.padding_feature = try_cuda(torch.zeros(NUM_VIEWS, args.action_embed_size))
        self.padding_blank = try_cuda(torch.zeros(0, self.hidden_size))
        self.rav, self.rav_count = 0, 0
        self.rav_step, self.rav_count_step = [0] * self.episode_len * max_sub_tasks, \
                                             [0] * self.episode_len * max_sub_tasks
        self.max_sub_tasks = max_sub_tasks
    
    def reset_rav(self):
        self.rav, self.rav_count = 0, 0
        self.rav_step, self.rav_count_step = [0] * self.episode_len * self.max_sub_tasks, \
                                             [0] * self.episode_len * self.max_sub_tasks
        
    def write_results(self):
        with open(self.results_path, 'w') as f:
            pretty_json_dump(self.results, f)
    
    def rollout(self, batch, train=True, reward_flag=False, use_loss=False, history=False, reward_type='dis',
                exp_forget=0.5):
        if history:
            history_context = self.make_history_context(batch, decay=exp_forget)
        else:
            history_context = None
        
        seq, seq_mask, seq_lengths = self.proc_batch(batch)
        obs = self.env.reset(next_batch=batch)
        
        done = np.zeros(self.batch_size, dtype=np.uint8)
        self.loss = 0
        self.acc = 0
        count_valid = 0
        count_steps = 0
        
        # Record starting point
        traj = [{'instr_id': ob['instr_id'],
                 'scores': [],
                 'heading': ob['heading'],
                 'trajectory': [ob['viewpoint']],
                 'trajectory_radians': [(ob['heading'], ob['elevation'])],
                 'reward': [], } for ob in obs]
        
        # Init text embed and action
        ctx, h_t, c_t = self.encoder(seq, seq_lengths, int(max(seq_lengths)))
        u_t_prev = self.padding_action.expand(self.batch_size, -1)
        
        # Do a sequence rollout and calculate the loss
        sequence_scores = try_cuda(torch.zeros(self.batch_size))
        for t in range(self.episode_len):
            f_t = self.feature_variables(obs)
            all_u_t, action_mask = self.action_variable(obs)
            h_t, c_t, alpha, logit, alpha_v = self.decoder(u_t_prev, all_u_t, f_t, h_t, c_t, ctx,
                                                           ctx_mask=seq_mask, history_context=history_context)
            
            # Supervised training
            logit[action_mask] = -float('inf')
            target = self.teacher_action(obs, done)
            
            if torch.isnan(logit).sum():
                print("Error: network produce nan result!")
                exit(0)
            
            # Determine next model inputs
            if self.feedback == 'teacher':
                a_t = torch.clamp(target, min=0)
            elif self.feedback == 'argmax':
                _, a_t = logit.max(1)
                a_t = a_t.detach()
            elif self.feedback == 'sample':
                probs = F.softmax(logit, dim=1)
                probs[action_mask] = 0
                m = D.Categorical(probs)
                a_t = m.sample()
            elif self.feedback == 'stochastic':
                a_t1 = torch.clamp(target, min=0)
                probs = F.softmax(logit, dim=1)
                probs[action_mask] = 0
                m = D.Categorical(probs)
                a_t2 = m.sample()
                selected_ind = try_cuda(torch.from_numpy(np.random.rand(self.batch_size) > 0.25).long())
                a_t = torch.cat((a_t1.unsqueeze(1), a_t2.unsqueeze(1)), dim=1).gather(1, selected_ind.unsqueeze(
                    1)).squeeze(1)
            else:
                sys.exit('Invalid feedback option')
            
            # Valid acc count
            count_valid += len(obs) - done.sum()
            count_steps += 1
            
            # Update the previous action
            u_t_prev = all_u_t[np.arange(self.batch_size), a_t, :].detach()
            action_scores = -F.cross_entropy(logit, a_t.clone(), ignore_index=-1, reduction='none')
            action_scores[done] = 0
            sequence_scores += action_scores
            
            # Calculate loss
            if use_loss:
                mask_done = try_cuda(torch.from_numpy(1 - done).byte())
                self.loss += self.criterion(logit[mask_done], target[mask_done])
            
            # Make environment action
            a_t[done] = 0
            obs, reward, next_done = self.env.step(obs, actions=a_t.tolist(), reward_type=reward_type)
            
            # Save trajectory output
            for i, ob in enumerate(obs):
                if not done[i]:
                    if reward_flag:
                        traj[i]['reward'].append(reward[i])
                        traj[i]['scores'].append(-action_scores[i])
                    traj[i]['trajectory'].append(ob['viewpoint'])
                    traj[i]['trajectory_radians'].append((ob['heading'], ob['elevation']))
            
            # Early exit if all ended
            done = next_done
            if done.all():
                break
        
        for i, ob in enumerate(obs):
            traj[i]['score'] = sequence_scores[i].item() / len(traj[i]['trajectory'])
            self.acc += (traj[i]['trajectory'][-1] == batch[i]['path'][-1]) / self.batch_size
            if train:
                traj[i]['score_loss'] = -sequence_scores[i]
            if reward_type == 'dtw' and reward_flag:
                traj[i]['reward'][-1] = self.env.get_dtw(self.env.batch[i]['scan'], self.env.batch[i]['path'],
                                                         traj[i]['trajectory'])
            elif reward_type != 'dis' and reward_flag:
                traj[i]['reward'][-1] = self.env.get_cls(self.env.batch[i]['scan'], self.env.batch[i]['path'],
                                                         traj[i]['trajectory'])
        return traj
    
    def make_history_context(self, batch, decay=0.5):
        history_lengths = [len(b['history_heading']) for b in batch]
        max_history = max(history_lengths)
        context_list = []
        text_context_list = []
        for hist_count in range(max_history):
            new_batch = copy.deepcopy(batch)
            zero_list = []
            for i, b in enumerate(new_batch):
                if len(b['history_heading']) > hist_count:
                    b['heading'] = b['history_heading'][hist_count]
                    b['path'] = b['history_path'][hist_count]
                    b['instr_encoding'] = b['history_instr_encoding'][hist_count]
                else:
                    b['path'] = [b['path'][0]]
                    b['instr_encoding'] = np.array([vocab_eos_idx])
                    zero_list.append(i)
            path_obs, path_actions, encoded_instructions = self.env.gold_obs_actions_and_instructions(new_batch)
            batched_image_features, batched_action_embeddings, seq_lengths = \
                self.batch_observations_and_actions(path_obs, path_actions)
            seq_lengths[zero_list] = 0
            context = self.decoder(batched_image_features, batched_action_embeddings, seq_lengths,
                                   int(max(seq_lengths)), context=True)
            context_list.append(context)
            max_len = max([len(ins) for ins in encoded_instructions])
            batched_ins, _, ins_lengths = self.batch_instructions_from_encoded(encoded_instructions, max_len + 2,
                                                                               cut=False)
            text_context = self.encoder(batched_ins, ins_lengths, context=True)
            text_context_list.append(text_context)
        
        context_list = torch.stack(context_list, dim=1) if context_list else []
        text_context_list = torch.stack(text_context_list, dim=1) if text_context_list else []
        if len(context_list) > 0:
            exp_weight = np.zeros((len(history_lengths), max_history))
            for i, h in enumerate(history_lengths):
                exp_weight[i][:h] = [np.exp(-x * decay) for x in range(h)][::-1]
            exp_weight = F.normalize(try_cuda(torch.from_numpy(exp_weight)).float(), p=1, dim=1).unsqueeze(-1)
            context = (context_list * exp_weight).sum(dim=1)
            text_context = (text_context_list * exp_weight).sum(dim=1)
        else:
            context = try_cuda(torch.zeros(len(history_lengths), self.hidden_size))
            text_context = try_cuda(torch.zeros(len(history_lengths), self.hidden_size))
        return [context, text_context]
    
    def proc_batch(self, batch, key='instr_encoding'):
        encoded_instructions = [item[key] for item in batch]
        return self.batch_instructions_from_encoded(encoded_instructions, self.max_instruction_length,
                                                    reverse=self.reverse_instruction)
    
    def batch_instructions_from_encoded(self, encoded_instructions, max_length, reverse=False, cut=True):
        num_instructions = len(encoded_instructions)
        seq_tensor = np.full((num_instructions, max_length), -1)
        seq_lengths = []
        for i, inst in enumerate(encoded_instructions):
            if len(inst) > 0 and inst[-1] == vocab_eos_idx:
                inst = inst[:-1]
            if reverse:
                inst = inst[::-1]
            inst = np.concatenate((inst, [vocab_eos_idx]))
            inst = inst[:max_length]
            seq_tensor[i, :len(inst)] = inst
            seq_lengths.append(len(inst))
        
        if cut:
            seq_tensor = torch.from_numpy(seq_tensor)[:, :max(seq_lengths)]
            mask = (seq_tensor == -1)[:, :max(seq_lengths)]
        else:
            seq_tensor = torch.from_numpy(seq_tensor)
            mask = (seq_tensor == -1)
        seq_tensor[mask] = vocab_pad_idx
        
        ret_tp = try_cuda(Variable(seq_tensor, requires_grad=False).long()), \
                 try_cuda(mask.byte()), \
                 try_cuda(torch.tensor(seq_lengths).long())
        return ret_tp
    
    def batch_observations_and_actions(self, path_obs, path_actions):
        seq_lengths = np.array([len(a) for a in path_actions])
        max_path_length = seq_lengths.max()
        image_features = [[] for _ in range(self.batch_size)]
        action_embeddings = [[] for _ in range(self.batch_size)]
        for i in range(self.batch_size):
            assert len(path_obs[i]) == len(path_actions[i])
            image_features[i] = [ob['feature'][0] for ob in path_obs[i]]
            action_embeddings[i] = [ob['action_embedding'][path_actions[i][j]] for j, ob in enumerate(path_obs[i])]
            image_features[i].extend([self.padding_feature] * (max_path_length - len(path_actions[i])))
            action_embeddings[i].extend([self.padding_action] * (max_path_length - len(path_actions[i])))
            image_features[i] = torch.stack(image_features[i], dim=0)
            action_embeddings[i] = torch.stack(action_embeddings[i], dim=0)
        batched_image_features = torch.stack(image_features, dim=0)
        batched_action_embeddings = torch.stack(action_embeddings, dim=0)
        return batched_image_features, batched_action_embeddings, try_cuda(torch.tensor(seq_lengths).long())
    
    def feature_variables(self, obs):
        ''' Extract precomputed features into variable. '''
        feature_lists = list(zip(*[ob['feature'] for ob in obs]))
        return torch.stack(feature_lists[0])
    
    def action_variable(self, obs):
        ''' Get the available action embedding for the agent to select.'''
        max_num_a = max([len(ob['adj_loc_list']) for ob in obs])
        is_valid = np.zeros((self.batch_size, max_num_a), np.float32)
        action_embeddings = []
        for i, ob in enumerate(obs):
            is_valid[i, len(ob['adj_loc_list']):] = 1
            action_embeddings.append(ob['action_embedding'])
        return pad_sequence(action_embeddings, batch_first=True), try_cuda(
            Variable(torch.from_numpy(is_valid).byte(), requires_grad=False))
    
    def teacher_action(self, obs, ended):
        ''' Extract teacher actions into variable. '''
        a = torch.LongTensor(len(obs))
        for i, ob in enumerate(obs):
            a[i] = ob['teacher_action'] if not ended[i] else -1
        return try_cuda(Variable(a, requires_grad=False))
    
    def criterion(self, logit, target):
        return F.cross_entropy(logit, target, ignore_index=-1)
    
    def train_reward(self, batch, n_iters, speaker, feedback='sample', beam_size=8, history=False, reward_type='dis',
                     exp_forget=0.5):
        self.feedback = feedback
        self.encoder.train()
        self.decoder.train()
        self.losses = []
        self.acces = []
        self.rewards = []
        self.ext_rewards = []
        delta = 0.5
        gamma = 0.95
        for i in range(n_iters):
            self.encoder_optimizer.zero_grad()
            self.decoder_optimizer.zero_grad()
            follower_results_cache = []
            reward_cache = []
            ext_reward_cache = []
            for beam in range(beam_size):
                follower_results = self.rollout(batch[i], reward_flag=True, history=history, reward_type=reward_type,
                                                exp_forget=exp_forget)
                ext_reward_cache.append([f_result['reward'][-1] for f_result in follower_results])
                if speaker is not None:
                    speaker_results = speaker.query(batch[i], follower_results, feedback='teacher', train=False,
                                                    original=False)
                    reward_cache.append([s_result['score'] * delta for s_result in speaker_results])
                else:
                    reward_cache.append([0] * self.batch_size)
                follower_results_cache.append(follower_results)
                self.acces.append(self.acc)
            self.loss = 0
            
            adv_array = np.zeros((beam_size, self.batch_size, self.episode_len))
            for beam in range(beam_size):
                R_intr = reward_cache[beam]
                for idx, f_result in enumerate(follower_results_cache[beam]):
                    a_len = len(f_result['reward'])
                    R_extr = 0
                    for j in range(a_len):
                        R_extr = R_extr * gamma + f_result['reward'][a_len - 1 - j]
                        Adv = (R_extr + R_intr[idx])
                        adv_array[beam, idx, j] = Adv
                        self.rav_step[j] = (self.rav_step[j] * self.rav_count_step[j] + Adv) / (
                            self.rav_count_step[j] + 1)
                        self.rav_count_step[j] += 1
            
            for beam in range(beam_size):
                for idx, f_result in enumerate(follower_results_cache[beam]):
                    a_len = len(f_result['reward'])
                    for j in range(a_len):
                        self.loss += (adv_array[beam, idx, j] - self.rav_step[j]) * f_result['scores'][a_len - 1 - j]
            self.rewards.append(reward_cache)
            self.ext_rewards.append(ext_reward_cache)
            self.loss /= (self.batch_size * beam_size)
            self.loss.backward()
            self.losses.append(self.loss.item())
            torch.nn.utils.clip_grad_norm_(self.encoder.parameters(), 20.)
            torch.nn.utils.clip_grad_norm_(self.decoder.parameters(), 20.)
            self.encoder_optimizer.step()
            self.decoder_optimizer.step()
    
    def train_crl(self, batch, n_iters, speaker, feedback='sample', beam_size=8, history=True, curriculum=0,
                  reward_type='dis', exp_forget=0.5):
        self.feedback = feedback
        self.encoder.train()
        self.decoder.train()
        self.losses = []
        self.acces = []
        self.rewards = []
        self.ext_rewards = []
        delta = 0.5
        for i in range(n_iters):
            self.encoder_optimizer.zero_grad()
            self.decoder_optimizer.zero_grad()
            follower_results_cache = []
            reward_cache = []
            ext_reward_cache = []
            for beam in range(beam_size):
                follower_results = self.reward_rollout(batch[i], curriculum + 1, history=history,
                                                       reward_type=reward_type, exp_forget=exp_forget)
                follower_results_cache.append(follower_results)
                ext_reward_cache.append([f_result['reward'][0] for f_result in follower_results])
                if speaker is not None:
                    speaker_results = speaker.query(batch[i], follower_results, feedback='teacher', train=False,
                                                    original=True, curriculum=True)
                    reward_cache.append([s_result['score'] * delta for s_result in speaker_results])
                else:
                    reward_cache.append([0] * self.batch_size)
                self.acces.append(self.acc)
            
            self.loss = 0
            adv_array = np.zeros((beam_size, self.batch_size, self.episode_len * self.max_sub_tasks))
            for beam in range(beam_size):
                R_intr = reward_cache[beam]
                for idx, f_result in enumerate(follower_results_cache[beam]):
                    a_len = len(f_result['scores'])
                    for j in range(a_len):
                        Adv = f_result['reward'][j] + R_intr[idx]
                        adv_array[beam, idx, j] = Adv
                        self.rav_step[j] = (self.rav_step[j] * self.rav_count_step[j] + Adv) / \
                                           (self.rav_count_step[j] + 1)
                        self.rav_count_step[j] += 1
            
            for beam in range(beam_size):
                for idx, f_result in enumerate(follower_results_cache[beam]):
                    a_len = len(f_result['scores'])
                    for j in range(a_len):
                        self.loss += float(adv_array[beam, idx, j] - self.rav_step[j]) * f_result['scores'][j]
            
            self.rewards.append(reward_cache)
            self.ext_rewards.append(ext_reward_cache)
            self.loss /= (self.batch_size * beam_size)
            self.loss.backward()
            self.losses.append(self.loss.item())
            torch.nn.utils.clip_grad_norm_(self.encoder.parameters(), 20.)
            torch.nn.utils.clip_grad_norm_(self.decoder.parameters(), 20.)
            self.encoder_optimizer.step()
            self.decoder_optimizer.step()
    
    def reward_rollout(self, batch, count, history=True, reward_type='dis', exp_forget=0.5):
        # count is total steps
        gamma = 0.95
        start_idx = [len(item) - count if len(item) >= count else 0 for item in batch]
        start_point = [item[start_idx[j]]['path'][0] for j, item in enumerate(batch)]
        rollout_results = [{'instr_id': item[0]['instr_id'],
                            'trajectory': [start_point[j]],
                            'reward': [],
                            'scores': []} for j, item in enumerate(batch)]
        if history:
            start_heading = [item[start_idx[j]]['history_heading'] for j, item in enumerate(batch)]
            start_path = [item[start_idx[j]]['history_path'] for j, item in enumerate(batch)]
            history_heading, history_path = [[] for _ in range(self.batch_size)], [[] for _ in range(self.batch_size)]
        for i in range(count):
            current_batch = copy.deepcopy([(b[start_idx[j] + i] if i <= (len(b) - 1) else b[-1])
                                           for j, b in enumerate(batch)])
            for j, b in enumerate(current_batch):
                b['path'] = [start_point[j], batch[j][-1]['path'][-1]]  # start to use final goal as objective
                if i <= len(batch[j]) - 1 and history:
                    b['history_heading'] = start_heading[j] + history_heading[j]
                    b['history_path'] = start_path[j] + history_path[j]
            follower_results = self.rollout(current_batch, reward_flag=True, history=history, reward_type=reward_type,
                                            exp_forget=exp_forget)
            start_point = [f_result['trajectory'][-1] for f_result in follower_results]
            for j, f_result in enumerate(follower_results):
                if start_idx[j] + i <= len(batch[j]) - 1:
                    rollout_results[j]['trajectory'].extend(f_result['trajectory'][1:])
                    rollout_results[j]['scores'].extend(f_result['scores'])
                    if history:
                        history_heading[j].append(f_result['heading'])
                        history_path[j].append(f_result['trajectory'])
                        if rollout_results[j]['trajectory'][-1] == rollout_results[j]['trajectory'][-2]:
                            rollout_results[j]['trajectory'] = rollout_results[j]['trajectory'][:-1]
                            history_path[j][-1] = history_path[j][-1][:-1]
                    if start_idx[j] + i != len(batch[j]) - 1:
                        f_result['reward'][-1] = 0
                    rollout_results[j]['reward'].extend(f_result['reward'])
        
        for j, r_result in enumerate(rollout_results):
            # trajectory
            b = batch[j][start_idx[j]]
            path = []
            for p in b['history_path']:
                path.extend(p[:-1])
            total_path = []
            for p in batch[j][-1]['history_path']:
                total_path.extend(p[:-1])
            total_path.extend(batch[j][-1]['path'])
            r_result['trajectory'] = path + r_result['trajectory']
            # reward
            r_result['reward'] = r_result['reward'][::-1]
            cls = self.env.get_cls(batch[j][-1]['scan'], total_path, rollout_results[j]['trajectory'])
            r_result['reward'][0] = cls
            R_cum = 0
            for k, r in enumerate(r_result['reward']):
                R_cum = R_cum * gamma + r
                r_result['reward'][k] = R_cum
            r_result['scores'] = r_result['scores'][::-1]
        return rollout_results
    
    def train(self, batch, n_iters, feedback='sample', history=False, exp_forget=0.5):
        ''' Train for a given number of iterations '''
        self.feedback = feedback
        self.encoder.train()
        self.decoder.train()
        self.losses = []
        self.acces = []
        for i in range(n_iters):
            self.encoder_optimizer.zero_grad()
            self.decoder_optimizer.zero_grad()
            self.rollout(batch[i], use_loss=True, history=history, exp_forget=exp_forget)
            self.loss.backward()
            self.losses.append(self.loss.item())
            self.acces.append(self.acc)
            torch.nn.utils.clip_grad_norm_(self.encoder.parameters(), 20.)
            torch.nn.utils.clip_grad_norm_(self.decoder.parameters(), 20.)
            self.encoder_optimizer.step()
            self.decoder_optimizer.step()
    
    def test(self, batch, feedback='argmax', one_by_one=False, history=False, exp_forget=0.5):
        ''' Evaluate once on each instruction in the current environment '''
        with torch.no_grad():
            self.feedback = feedback
            self.encoder.eval()
            self.decoder.eval()
            
            self.results = {}
            count = 0
            looped = False
            while True:
                if one_by_one:
                    max_executes = max([len(b) for b in batch[count]])
                    rollout_results = [{'instr_id': item[0]['instr_id'],
                                        'trajectory': [item[0]['path'][0]],
                                        'trajectory_radians': [(item[0]['heading'], 0.0)],
                                        'segments': len(item)} for item in batch[count]]
                    start_point = [item[0]['path'][0] for item in batch[count]]
                    start_heading = [item[0]['heading'] for item in batch[count]]
                    history_heading, history_path, history_instr, history_instr_encoding = \
                        [[] for _ in range(self.batch_size)], [[] for _ in range(self.batch_size)], \
                        [[] for _ in range(self.batch_size)], [[] for _ in range(self.batch_size)]
                    for i in range(max_executes):
                        current_batch = copy.deepcopy([(b[i] if i < len(b) else b[-1]) for b in batch[count]])
                        for j, b in enumerate(current_batch):
                            b['path'] = [start_point[j]]
                            b['heading'] = start_heading[j]
                            if history and i < len(batch[count][j]):
                                b['history_heading'] = history_heading[j]
                                b['history_path'] = history_path[j]
                                b['history_instr'] = history_instr[j]
                                b['history_instr_encoding'] = history_instr_encoding[j]
                        follower_results = self.rollout(current_batch, train=False, history=history,
                                                        exp_forget=exp_forget)
                        start_point = [f_result['trajectory'][-1] for f_result in follower_results]
                        start_heading = [f_result['trajectory_radians'][-1][0] for f_result in follower_results]
                        for j, f_result in enumerate(follower_results):
                            if i >= len(batch[count][j]):
                                continue
                            rollout_results[j]['trajectory'].extend(f_result['trajectory'][1:])
                            rollout_results[j]['trajectory_radians'].extend(f_result['trajectory_radians'][1:])
                            if rollout_results[j]['trajectory'][-1] == rollout_results[j]['trajectory'][-2] and \
                                i < len(batch[count][j]) - 1:
                                rollout_results[j]['trajectory'] = rollout_results[j]['trajectory'][:-1]
                                rollout_results[j]['trajectory_radians'] = rollout_results[j]['trajectory_radians'][:-1]
                            if history and i < len(batch[count][j]):
                                history_heading[j].append(f_result['heading'])
                                history_path[j].append(f_result['trajectory'][:-1])
                                history_instr[j].append(current_batch[j]['instructions'])
                                history_instr_encoding[j].append(current_batch[j]['instr_encoding'])
                else:
                    rollout_results = self.rollout(batch[count], train=False, history=history, exp_forget=exp_forget)
                
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
        return base_path + "_enc", base_path + "_dec", base_path + "_enc_opt", base_path + "_dec_opt"
    
    def save(self, path):
        ''' Snapshot models '''
        encoder_path, decoder_path, encoder_opt_path, decoder_opt_path = self.encoder_and_decoder_paths(path)
        torch.save(self.encoder.state_dict(), encoder_path)
        torch.save(self.decoder.state_dict(), decoder_path)
        torch.save(self.encoder_optimizer.state_dict(), encoder_opt_path)
        torch.save(self.decoder_optimizer.state_dict(), decoder_opt_path)
    
    def load(self, path, load_opt, **kwargs):
        ''' Loads parameters (but not training state) '''
        encoder_path, decoder_path, encoder_opt_path, decoder_opt_path = self.encoder_and_decoder_paths(path)
        self.encoder.load_my_state_dict(torch.load(encoder_path, **kwargs))
        self.decoder.load_my_state_dict(torch.load(decoder_path, **kwargs))
        if load_opt:
            self.encoder_optimizer.load_state_dict(torch.load(encoder_opt_path, **kwargs))
            self.decoder_optimizer.load_state_dict(torch.load(decoder_opt_path, **kwargs))
