import json
import sys
import numpy as np
import itertools
import torch
import copy
import torch.nn.functional as F
import torch.distributions as D

from src.vocab.tokenizer import VOCAB_PAD_IDX, VOCAB_EOS_IDX
from src.utils import batch_instructions_from_encoded, \
  batch_observations_and_actions
from model.cuda import try_cuda


class Seq2SeqSpeaker(object):
  def __init__(self, env, results_path, args, encoder, decoder, tok):
    self.env = env
    self.tok = tok
    self.results_path = results_path
    self.results = {}
    self.encoder = encoder
    self.decoder = decoder
    self.max_instruction_length = args.max_ins_len
  
  def write_results(self):
    with open(self.results_path, 'w') as f:
      json.dump(self.results, f)
  
  def _score_obs_actions_and_instructions(self, path_obs, path_actions,
                                          encoded_instructions, feedback,
                                          train):
    batch_size = len(path_obs)
    instr_seq, _, _ = \
      batch_instructions_from_encoded(encoded_instructions,
                                      self.max_instruction_length, cut=False)
    batched_image_features, batched_action_embeddings, path_mask, seq_len = \
      batch_observations_and_actions(path_obs, path_actions,
                                     self.env.padding_feature,
                                     self.env.padding_action)
    
    ctx = self.encoder(batched_image_features, batched_action_embeddings,
                       seq_len)
    h_t = try_cuda(torch.zeros(batch_size, ctx.size(-1)))
    c_t = try_cuda(torch.zeros(batch_size, ctx.size(-1)))
    ended = np.array([False] * batch_size)
    
    outputs = [{
      'instr_id': path_obs[i][0]['instr_id'],
      'word_indices': [],
      'scores': []
    } for i in range(batch_size)]
    
    # Do a sequence rollout and calculate the loss
    loss = 0
    w_t = try_cuda(torch.from_numpy(
      np.full((batch_size, 1), self.tok.vocab_bos_idx,
              dtype='int64')).long())
    
    if train:
      w_t = torch.cat([w_t, instr_seq], dim=1)
      logits, _, _ = self.decoder(w_t, ctx, path_mask, h_t, c_t)
      logits = logits.permute(0, 2, 1).contiguous()
      loss = F.cross_entropy(
        input=logits[:, :, :-1],  # -1 for aligning
        target=instr_seq,  # "1:" to ignore the word <BOS>
        ignore_index=VOCAB_PAD_IDX
      )
    else:
      sequence_scores = try_cuda(torch.zeros(batch_size))
      for t in range(self.max_instruction_length):
        logit, h_t, c_t = self.decoder(w_t.view(-1, 1), ctx, path_mask, h_t,
                                       c_t)
        logit = logit.squeeze(1)
        
        logit[:, VOCAB_PAD_IDX] = -float('inf')
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
          probs[:, VOCAB_PAD_IDX] = 0
          m = D.Categorical(probs)
          w_t = m.sample()
        else:
          sys.exit('Invalid feedback option')
        
        log_probs = F.log_softmax(logit, dim=1)
        word_scores = -F.nll_loss(log_probs, w_t, ignore_index=VOCAB_PAD_IDX,
                                  reduction='none')
        sequence_scores += word_scores
        loss += F.nll_loss(log_probs, target, ignore_index=VOCAB_PAD_IDX)
        
        for i in range(batch_size):
          word_idx = w_t[i].item()
          if not ended[i]:
            outputs[i]['word_indices'].append(int(word_idx))
            outputs[i]['scores'].append(word_scores[i].item())
          if word_idx == VOCAB_EOS_IDX:
            ended[i] = True
        
        # Early exit if all ended
        if ended.all():
          break
      
      for i, item in enumerate(outputs):
        item['score'] = float(sequence_scores[i].item()) / len(
          item['word_indices'])
        item['words'] = self.tok.decode_sentence(item['word_indices'],
                                                 break_on_eos=True, join=False)
    
    return outputs, loss
  
  def _rollout(self, batch, train=True):
    path_obs, path_actions, encoded_instructions = \
      self.env.gold_obs_actions_and_instructions(batch)
    outputs, loss = \
      self._score_obs_actions_and_instructions(path_obs, path_actions,
                                               encoded_instructions,
                                               self.feedback, train)
    return outputs
  
  def query(self, batch, follower_results, feedback='argmax',
            curriculum=False):
    self.feedback = feedback
    if not curriculum:
      next_batch = [copy.deepcopy(b) for b in batch]
    else:
      next_batch = [copy.deepcopy(b[-1]) for b in batch]
    for i, b in enumerate(next_batch):
      if 'history_path' in b and len(b['history_path']) > 0:
        b['path'] = follower_results[i]['trajectory']
        b['heading'] = b['history_heading'][0]
        b['instr_encoding'] \
          = list(itertools.chain.from_iterable(b['history_instr_encoding'])) \
            + list(b['instr_encoding'])
      else:
        b['path'] = follower_results[i]['trajectory']
    with torch.no_grad():
      self.encoder.eval()
      self.decoder.eval()
      results = self._rollout(next_batch, train=False)
      return results
  
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
        rollout_results = self._rollout(batch[count], train=False)
        count += 1
        
        for result in rollout_results:
          if result['instr_id'] in self.results:
            looped = True
          else:
            self.results[result['instr_id']] = result
        if looped or len(batch) == count:
          break
    return self.results
  
  def _encoder_and_decoder_paths(self, base_path):
    return base_path + "_enc", base_path + "_dec"
  
  def save(self, path):
    ''' Snapshot models '''
    encoder_path, decoder_path = self._encoder_and_decoder_paths(path)
    torch.save(self.encoder.state_dict(), encoder_path)
    torch.save(self.decoder.state_dict(), decoder_path)
  
  def load(self, path, **kwargs):
    ''' Loads parameters (but not training state) '''
    encoder_path, decoder_path = self._encoder_and_decoder_paths(path)
    self.encoder.load_my_state_dict(torch.load(encoder_path, **kwargs))
    self.decoder.load_my_state_dict(torch.load(decoder_path, **kwargs))
