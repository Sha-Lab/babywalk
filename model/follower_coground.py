import math
import torch
import torch.nn as nn

from torch.autograd import Variable
from cuda import try_cuda
from attentions import WhSoftDotAttention
from context_encoder import ContextEncoder


class PositionalEncoding(nn.Module):
  def __init__(self, d_model, dropout, max_len):
    super(PositionalEncoding, self).__init__()
    self.dropout = nn.Dropout(p=dropout)
    
    # Compute the PE once
    pe = torch.zeros(max_len, d_model)
    position = torch.arange(0, max_len).unsqueeze(1).float()
    div_term = torch.exp(
      torch.arange(0, d_model, 2).float() / d_model * (-math.log(10000.0))
    )
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    pe = pe.unsqueeze(0)
    self.register_buffer('pe', pe)
  
  def forward(self, x):
    x = x + Variable(self.pe[:, :x.size(1)], requires_grad=False)
    return self.dropout(x)


class CogroundDecoderLSTM(nn.Module):
  def __init__(self, embedding_size, hidden_size, dropout_ratio, feature_size,
               max_len, history=False, visual_hidden_size=1024):
    super(CogroundDecoderLSTM, self).__init__()
    self.embedding_size = embedding_size
    self.feature_size = feature_size
    self.hidden_size = hidden_size
    self.u_begin = try_cuda(Variable(torch.zeros(embedding_size),
                                     requires_grad=False))
    self.drop = nn.Dropout(p=dropout_ratio)
    self.lstm = nn.LSTMCell(2 * embedding_size + hidden_size, hidden_size)
    self.text_attention_layer = WhSoftDotAttention(hidden_size, hidden_size)
    self.positional_encoding = PositionalEncoding(hidden_size,
                                                  dropout=0, max_len=max_len)
    self.visual_attention_layer = WhSoftDotAttention(hidden_size,
                                                     visual_hidden_size)
    self.visual_mlp = nn.Sequential(
      nn.BatchNorm1d(feature_size),
      nn.Linear(feature_size, visual_hidden_size),
      nn.BatchNorm1d(visual_hidden_size),
      nn.Dropout(dropout_ratio),
      nn.ReLU()
    )
    self.action_attention_layer = WhSoftDotAttention(hidden_size * 2,
                                                     visual_hidden_size)
    self.sm = nn.Softmax(dim=1)
    if history:
      self.linear_context_out = nn.Linear(hidden_size, hidden_size, bias=True)
      self.linear_text_out = nn.Linear(hidden_size, hidden_size, bias=True)
      self.context_encoder = ContextEncoder(feature_size, hidden_size,
                                            dropout_ratio)
      self.linear_combine = nn.Sequential(
        nn.Linear(hidden_size * 4, hidden_size * 2, bias=True),
        nn.ReLU(),
        nn.Linear(hidden_size * 2, hidden_size * 2, bias=True)
      )
  
  def load_my_state_dict(self, state_dict):
    own_state = self.state_dict()
    for name, param in state_dict.items():
      if name not in own_state:
        continue
      if isinstance(param, nn.Parameter):
        param = param.data
      own_state[name].copy_(param)
  
  def forward(self, *args, **kwargs):
    if 'context' in kwargs and kwargs['context'] == True:
      return self.forward_context(*args)
    else:
      return self.forward_current(*args, ctx_mask=kwargs['ctx_mask'],
                                  history_context=kwargs['history_context'])
  
  def forward_current(self, u_t_prev, all_u_t, visual_context, h_0, c_0, ctx,
                      ctx_mask=None, history_context=None):
    '''
    u_t_prev: batch x embedding_size
    all_u_t: batch x a_num x embedding_size
    visual_context: batch x v_num x feature_size => panoramic view, DEP
    h_0: batch x hidden_size
    c_0: batch x hidden_size
    ctx: batch x seq_len x dim
    ctx_mask: batch x seq_len - indices to be masked
    '''
    ctx_pos = self.positional_encoding(ctx)
    attn_text, _alpha_text = \
      self.text_attention_layer(h_0, ctx_pos, v=ctx, mask=ctx_mask)
    alpha_text = self.sm(_alpha_text)
    
    batch_size, a_size, _ = all_u_t.size()
    g_v = all_u_t.view(-1, self.feature_size)
    g_v = self.visual_mlp(g_v).view(batch_size, a_size, -1)
    attn_vision, _alpha_vision = \
      self.visual_attention_layer(h_0, g_v, v=all_u_t)
    alpha_vision = self.sm(_alpha_vision)
    
    concat_input = torch.cat((attn_text, attn_vision, u_t_prev), 1)
    drop = concat_input
    h_1, c_1 = self.lstm(drop, (h_0, c_0))
    
    if history_context is not None:
      context = self.linear_context_out(history_context[0])
      text = self.linear_text_out(history_context[1])
      action_selector = self.linear_combine(
        torch.cat((attn_text, h_1, context, text), 1))
    else:
      action_selector = torch.cat((attn_text, h_1), 1)
    _, alpha_action = self.action_attention_layer(action_selector, g_v)
    return h_1, c_1, alpha_text, alpha_action, alpha_vision
  
  def forward_context(self, *args):
    return self.context_encoder(*args)
