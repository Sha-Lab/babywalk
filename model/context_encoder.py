import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from attentions import SoftDotAttention
from cuda import try_cuda


class ContextEncoder(nn.Module):
  def __init__(self, feature_size, hidden_size, dropout_ratio):
    ''' Bidirectional LSTM ContextEncoder. '''
    super().__init__()
    self.hidden_size = hidden_size
    self.feature_size = feature_size
    self.lstm = nn.LSTM(feature_size, self.hidden_size // 2, 1,
                        batch_first=True, bidirectional=True)
    self.drop = nn.Dropout(p=dropout_ratio)
    self.attention_layer = SoftDotAttention(self.hidden_size,
                                            ctx_dim=feature_size)
    self.post_lstm = nn.LSTM(self.hidden_size, self.hidden_size // 2, 1,
                             batch_first=True, bidirectional=True)
  
  def forward(self, feature, action_embeds, lengths):
    '''
    :param action_embeds: (batch_size, length, 2048). The feature of the view
    :param feature: (batch_size, length, 36, 2048). The action taken (with the image feature)
    :param lengths:
    :return: context with shape (batch_size, length, hidden_size) -> (batch_size, hidden_size)
    '''
    
    # LSTM on the action embed
    new_lengths = [1 if l == 0 else l for l in lengths]
    packed_embeds = pack_padded_sequence(action_embeds, new_lengths,
                                         enforce_sorted=False,
                                         batch_first=True)
    
    enc_h, _ = self.lstm(packed_embeds)
    ctx, _ = pad_packed_sequence(enc_h, batch_first=True)
    ctx = self.drop(ctx)
    
    # Att and Handle with the shape
    batch_size, max_length, _ = ctx.size()
    x, _ = self.attention_layer(  # Attend to the feature map
      ctx.contiguous().view(-1, self.hidden_size),
      # (batch, length, hidden) --> (batch x length, hidden)
      feature.view(batch_size * max_length, -1, self.feature_size),
      # (batch, length, # of images, feature_size) --> (batch x length, # of images, feature_size)
    )
    x = x.view(batch_size, max_length, -1)
    x = self.drop(x)
    
    # Post LSTM layer
    packed_x = pack_padded_sequence(x, new_lengths,
                                    enforce_sorted=False, batch_first=True)
    # self.post_lstm.flatten_parameters()
    enc_x, _ = self.post_lstm(packed_x)
    x, _ = pad_packed_sequence(enc_x, batch_first=True)
    
    out = torch.stack([x[i, l - 1, :] if l > 0
                       else try_cuda(torch.zeros(self.hidden_size))
                       for i, l in enumerate(lengths)], dim=0)
    return out


class LSTMMemory(nn.Module):
  def __init__(self, hidden_size):
    '''Initialize layer.'''
    super().__init__()
    self.hidden_size = hidden_size
    self.lstm = nn.LSTM(self.hidden_size, self.hidden_size,
                        batch_first=True, bidirectional=False)
  
  def forward(self, hidden_variables, lengths):
    if len(hidden_variables) != 0:
      x = hidden_variables  # [batch * length * hidden]
      ctx, _ = self.lstm(x)
      out = torch.stack([ctx[i, l - 1, :] if l > 0
                         else try_cuda(torch.zeros(self.hidden_size))
                         for i, l in enumerate(lengths)], dim=0)
      return out
    else:
      return try_cuda(torch.zeros(len(lengths), self.hidden_size))
