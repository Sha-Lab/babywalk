import torch
import torch.nn as nn
from attentions import SoftDotAttention


class SpeakerEncoderLSTM(nn.Module):
  def __init__(self, feature_size, hidden_size, dropout_ratio):
    ''' Bidirectional LSTM Speaker'''
    super().__init__()
    self.hidden_size = hidden_size
    self.feature_size = feature_size
    self.lstm = nn.LSTM(feature_size, self.hidden_size // 2, 1,
                        batch_first=True,
                        bidirectional=True)
    self.drop = nn.Dropout(p=dropout_ratio)
    self.drop3 = nn.Dropout(p=0.3)
    self.attention_layer = SoftDotAttention(self.hidden_size, feature_size)
    self.post_lstm = nn.LSTM(self.hidden_size, self.hidden_size // 2, 1,
                             batch_first=True,
                             bidirectional=True)
  
  def load_my_state_dict(self, state_dict):
    own_state = self.state_dict()
    for name, param in state_dict.items():
      if name not in own_state:
        continue
      if isinstance(param, nn.Parameter):
        param = param.data
      own_state[name].copy_(param)
  
  def forward(self, feature, action_embeds, lengths):
    """
    :param action_embeds: (batch_size, length, 2052). The feature of the view
    :param feature: (batch_size, length, 36, 2052). The action taken (with the image feature)
    :param lengths: Not used in it
    :return: context with shape (batch_size, length, hidden_size)
    """
    x = action_embeds
    x[..., :-128] = self.drop3(
      x[..., :-128])  # Do not dropout the spatial features
    
    # LSTM on the action embed
    ctx, _ = self.lstm(x)
    ctx = self.drop(ctx)
    
    # Att and Handle with the shape
    batch_size, max_length, _ = ctx.size()
    feature[..., :-128] = self.drop3(
      feature[..., :-128])  # Dropout the image feature
    x, _ = self.attention_layer(  # Attend to the feature map
      ctx.contiguous().view(-1, self.hidden_size),
      # (batch, length, hidden) --> (batch x length, hidden)
      feature.view(batch_size * max_length, -1, self.feature_size),
      # (batch, length, # of images, feature_size) --> (batch x length, # of images, feature_size)
    )
    x = x.view(batch_size, max_length, -1)
    x = self.drop(x)
    
    # Post LSTM layer
    x, _ = self.post_lstm(x)
    x = self.drop(x)
    
    return x


class SpeakerDecoderLSTM(nn.Module):
  def __init__(self, vocab_size, embedding_size, padding_idx, hidden_size,
               dropout_ratio, glove=None):
    super().__init__()
    self.hidden_size = hidden_size
    self.embedding = torch.nn.Embedding(vocab_size, embedding_size,
                                        padding_idx)
    self.use_glove = glove is not None
    if self.use_glove:
      print('Using GloVe embedding')
      self.embedding.weight.data[...] = torch.from_numpy(glove)
      self.embedding.weight.requires_grad = False
    self.lstm = nn.LSTM(embedding_size, hidden_size, batch_first=True)
    self.drop = nn.Dropout(dropout_ratio)
    self.attention_layer = SoftDotAttention(hidden_size, hidden_size)
    self.projection = nn.Linear(hidden_size, vocab_size)
    self.baseline_projection = nn.Sequential(
      nn.Linear(hidden_size, 128),
      nn.ReLU(),
      nn.Dropout(dropout_ratio),
      nn.Linear(128, 1)
    )
  
  def load_my_state_dict(self, state_dict):
    own_state = self.state_dict()
    for name, param in state_dict.items():
      if name not in own_state:
        continue
      if isinstance(param, nn.Parameter):
        param = param.data
      own_state[name].copy_(param)
  
  def forward(self, words, ctx, ctx_mask, h0, c0):
    h0, c0 = h0.unsqueeze(0), c0.unsqueeze(0)
    embeds = self.embedding(words)
    embeds = self.drop(embeds)
    x, (h1, c1) = self.lstm(embeds, (h0, c0))
    
    x = self.drop(x)
    
    # Get the size
    batchXlength = words.size(0) * words.size(1)
    multiplier = batchXlength // ctx.size(0)
    
    # Att and Handle with the shape
    # Reshaping x          <the output> --> (b(word)*l(word), r)
    # Expand the ctx from  (b, a, r)    --> (b(word)*l(word), a, r)
    # Expand the ctx_mask  (b, a)       --> (b(word)*l(word), a)
    x, _ = self.attention_layer(
      x.contiguous().view(batchXlength, self.hidden_size),
      ctx.unsqueeze(1).expand(-1, multiplier, -1, -1).contiguous().view(
        batchXlength, -1, self.hidden_size),
      mask=ctx_mask.unsqueeze(1).expand(-1, multiplier, -1).contiguous().view(
        batchXlength, -1)
    )
    x = x.view(words.size(0), words.size(1), self.hidden_size)
    
    # Output the prediction logit
    x = self.drop(x)
    logit = self.projection(x)
    
    return logit, h1.squeeze(0), c1.squeeze(0)
