import json
import sys
import random
import numpy as np
from collections import namedtuple
import time
import itertools

import torch
import copy
from torch.autograd import Variable
import torch.nn.functional as F
import torch.distributions as D

from src.vocab.tokenizer import vocab_pad_idx, vocab_eos_idx
from model.cuda import try_cuda

NUM_VIEWS = 36


class Seq2SeqSpeaker(object):
    def __init__(self, env, results_path, args, encoder, decoder, tok):
        self.env = env
        self.tok = tok
        self.batch_size = args.batch_size
        self.results_path = results_path
        self.results = {}
        self.encoder = encoder
        self.decoder = decoder
        self.max_instruction_length = args.max_ins_len
        self.padding_action = try_cuda(torch.zeros(args.action_embed_size))
    
    def write_results(self):
        with open(self.results_path, 'w') as f:
            json.dump(self.results, f)
    
    def batch_instructions_from_encoded(self, encoded_instructions, max_length, reverse=False):
        num_instructions = len(encoded_instructions)
        seq_tensor = np.full((num_instructions, max_length), vocab_pad_idx)
        seq_lengths = []
        for i, inst in enumerate(encoded_instructions):
            if len(inst) > 0:
                assert inst[-1] != vocab_eos_idx
            if reverse:
                inst = inst[::-1]
            inst = inst[:max_length - 1]
            inst = np.concatenate((inst, [vocab_eos_idx]))
            seq_tensor[i, :len(inst)] = inst
            seq_lengths.append(len(inst))
        
        seq_tensor = torch.from_numpy(seq_tensor)
        mask = (seq_tensor == vocab_pad_idx)[:, :max(seq_lengths)]
        
        ret_tp = try_cuda(seq_tensor.long()), \
                 try_cuda(mask.byte()), \
                 seq_lengths
        return ret_tp
    
    def _batch_observations_and_actions(self, path_obs, path_actions, pad=0):
        seq_lengths = np.array([len(a) for a in path_actions])
        max_path_length = seq_lengths.max()
        mask = np.ones((self.batch_size, max_path_length), np.uint8)
        image_features = [[] for _ in range(self.batch_size)]
        action_embeddings = [[] for _ in range(self.batch_size)]
        for i in range(self.batch_size):
            assert len(path_obs[i]) == len(path_actions[i])
            mask[i, :len(path_actions[i])] = 0
            image_features[i] = [ob['feature'][0] for ob in path_obs[i]]
            action_embeddings[i] = [ob['action_embedding'][path_actions[i][j]] for j, ob in enumerate(path_obs[i])]
            image_features[i].extend([image_features[i][-1]] * (max_path_length - len(path_actions[i])))
            action_embeddings[i].extend([self.padding_action] * (max_path_length - len(path_actions[i])))
            image_features[i] = torch.stack(image_features[i], dim=0)
            action_embeddings[i] = torch.stack(action_embeddings[i], dim=0)
        batched_image_features = torch.stack(image_features, dim=0)
        batched_action_embeddings = torch.stack(action_embeddings, dim=0)
        mask = try_cuda(torch.from_numpy(mask).byte())
        if pad > 0:
            batched_image_features = F.pad(batched_image_features, (0, 0, 0, 0, 0, pad - max_path_length))
            batched_action_embeddings = F.pad(batched_action_embeddings, (0, 0, 0, pad - max_path_length))
        return batched_image_features, batched_action_embeddings, mask, seq_lengths
    
    def _score_obs_actions_and_instructions(self, path_obs, path_actions, encoded_instructions, feedback, train):
        instr_seq, _, _ = self.batch_instructions_from_encoded(encoded_instructions, self.max_instruction_length)
        batched_image_features, batched_action_embeddings, path_mask, seq_len = \
            self._batch_observations_and_actions(path_obs, path_actions)
        
        ctx = self.encoder(batched_image_features, batched_action_embeddings, seq_len)
        h_t = try_cuda(torch.zeros(self.batch_size, ctx.size(-1)))
        c_t = try_cuda(torch.zeros(self.batch_size, ctx.size(-1)))
        
        ended = np.array([False] * self.batch_size)
        
        outputs = [{
            'instr_id': path_obs[i][0]['instr_id'],
            'word_indices': [],
            'scores': []
        } for i in range(self.batch_size)]
        
        # Do a sequence rollout and calculate the loss
        loss = 0
        w_t = try_cuda(Variable(torch.from_numpy(np.full((self.batch_size, 1), self.tok.vocab_bos_idx,
                                                         dtype='int64')).long()))
        
        if train:
            w_t = torch.cat([w_t, instr_seq], dim=1)
            logits, _, _ = self.decoder(w_t, ctx, path_mask, h_t, c_t)
            logits = logits.permute(0, 2, 1).contiguous()
            loss = F.cross_entropy(
                input=logits[:, :, :-1],  # -1 for aligning
                target=instr_seq,  # "1:" to ignore the word <BOS>
                ignore_index=vocab_pad_idx
            )
        else:
            sequence_scores = try_cuda(torch.zeros(self.batch_size))
            for t in range(self.max_instruction_length):
                logit, h_t, c_t = self.decoder(w_t.view(-1, 1), ctx, path_mask, h_t, c_t)
                logit = logit.squeeze(1)
                
                # BOS are not part of the encoded sequences
                # logit[:, vocab_unk_idx] = -float('inf')
                logit[:, vocab_pad_idx] = -float('inf')
                target = instr_seq[:, t].contiguous()
                
                if torch.isnan(logit).sum():
                    print("Error: network produce nan result!")
                    exit(0)
                
                # Determine next model inputs
                if feedback == 'teacher':
                    w_t = target
                elif feedback == 'argmax':
                    _, w_t = logit.max(1)
                    w_t = w_t.detach()
                elif feedback == 'sample':
                    probs = F.softmax(logit, dim=1)
                    probs[:, vocab_pad_idx] = 0
                    m = D.Categorical(probs)
                    w_t = m.sample()
                else:
                    sys.exit('Invalid feedback option')
                
                log_probs = F.log_softmax(logit, dim=1)
                word_scores = -F.nll_loss(log_probs, w_t, ignore_index=vocab_pad_idx, reduction='none')
                sequence_scores += word_scores
                loss += F.nll_loss(log_probs, target, ignore_index=vocab_pad_idx)
                
                for i in range(self.batch_size):
                    word_idx = w_t[i].item()
                    if not ended[i]:
                        outputs[i]['word_indices'].append(int(word_idx))
                        outputs[i]['scores'].append(word_scores[i].item())
                    if word_idx == vocab_eos_idx:
                        ended[i] = True
                
                # Early exit if all ended
                if ended.all():
                    break
            
            for i, item in enumerate(outputs):
                item['score'] = float(sequence_scores[i].item()) / len(item['word_indices'])
                item['words'] = self.tok.decode_sentence(item['word_indices'], break_on_eos=True, join=False)
        
        return outputs, loss
    
    def rollout(self, batch, train=True):
        path_obs, path_actions, encoded_instructions = \
            self.env.gold_obs_actions_and_instructions(batch)
        outputs, loss = self._score_obs_actions_and_instructions(path_obs, path_actions, encoded_instructions,
                                                                 self.feedback, train)
        return outputs
    
    def query(self, batch, follower_results, feedback='argmax', original=False, curriculum=False):
        self.feedback = feedback
        if not curriculum:
            next_batch = copy.deepcopy(batch)
        else:
            next_batch = copy.deepcopy([b[-1] for b in batch])
        for i, b in enumerate(next_batch):
            if original and 'history_path' in b and len(b['history_path']) > 0:
                b['path'] = follower_results[i]['trajectory']
                b['heading'] = b['history_heading'][0]
                b['instr_encoding'] = list(itertools.chain.from_iterable(b['history_instr_encoding'])) \
                                      + list(b['instr_encoding'])
            else:
                b['path'] = follower_results[i]['trajectory']
        with torch.no_grad():
            self.encoder.eval()
            self.decoder.eval()
            return self.rollout(next_batch, train=False)
    
    def test(self, batch, feedback='argmax'):
        ''' Evaluate once on each instruction in the current environment '''
        with torch.no_grad():
            self.feedback = feedback
            self.encoder.eval()
            self.decoder.eval()
            self.results = {}
            
            count = 0
            looped = False
            while True:
                rollout_results = self.rollout(batch[count], train=False)
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
        return base_path + "_enc", base_path + "_dec"
    
    def save(self, path):
        ''' Snapshot models '''
        encoder_path, decoder_path = self.encoder_and_decoder_paths(path)
        torch.save(self.encoder.state_dict(), encoder_path)
        torch.save(self.decoder.state_dict(), decoder_path)
    
    def load(self, path, **kwargs):
        ''' Loads parameters (but not training state) '''
        encoder_path, decoder_path = self.encoder_and_decoder_paths(path)
        self.encoder.load_my_state_dict(torch.load(encoder_path, **kwargs))
        self.decoder.load_my_state_dict(torch.load(decoder_path, **kwargs))
