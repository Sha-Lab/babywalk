import torch
import torch.nn as nn


class SoftDotAttention(nn.Module):
  '''Soft Dot Attention.

  Ref: http://www.aclweb.org/anthology/D15-1166
  Adapted from PyTorch OPEN NMT.
  '''
  
  def __init__(self, dim, ctx_dim=None):
    '''Initialize layer.'''
    super(SoftDotAttention, self).__init__()
    if ctx_dim is None:
      ctx_dim = dim
    self.linear_in = nn.Linear(dim, ctx_dim, bias=False)
    self.sm = nn.Softmax(dim=1)
    self.linear_out = nn.Linear(dim + ctx_dim, dim, bias=False)
    self.tanh = nn.Tanh()
  
  def forward(self, h, context, mask=None):
    '''Propagate h through the network.

    h: batch x dim
    context: batch x seq_len x dim
    mask: batch x seq_len indices to be masked
    '''
    target = self.linear_in(h).unsqueeze(2)  # batch x dim x 1
    
    # Get attention
    attn = torch.bmm(context, target).squeeze(2)  # batch x seq_len
    if mask is not None:
      # -Inf masking prior to the softmax
      attn.data.masked_fill_(mask, -float('inf'))
    attn = self.sm(attn)
    attn3 = attn.view(attn.size(0), 1, attn.size(1))  # batch x 1 x seq_len
    
    weighted_context = torch.bmm(attn3, context).squeeze(1)  # batch x dim
    h_tilde = torch.cat((weighted_context, h), 1)
    
    h_tilde = self.tanh(self.linear_out(h_tilde))
    return h_tilde, attn


class WhSoftDotAttention(nn.Module):
  ''' Visual Dot Attention Layer. '''
  
  def __init__(self, h_dim, v_dim=None):
    '''Initialize layer.'''
    super(WhSoftDotAttention, self).__init__()
    if v_dim is None:
      v_dim = h_dim
    self.h_dim = h_dim
    self.v_dim = v_dim
    self.linear_in_h = nn.Linear(h_dim, v_dim, bias=True)
    self.sm = nn.Softmax(dim=1)
  
  def forward(self, h, k, mask=None, v=None):
    '''Propagate h through the network.
    h: batch x h_dim
    k: batch x v_num x v_dim
    '''
    target = self.linear_in_h(h).unsqueeze(2)  # batch x dot_dim x 1
    attn = torch.bmm(k, target).squeeze(2)  # batch x v_num
    if mask is not None:
      attn.data.masked_fill_(mask, -float('inf'))
    attn_sm = self.sm(attn)
    attn3 = attn_sm.view(attn.size(0), 1, attn.size(1))  # batch x 1 x v_num
    ctx = v if v is not None else k
    weighted_context = torch.bmm(attn3, ctx).squeeze(1)  # batch x v_dim
    return weighted_context, attn


class TextDotAttention(nn.Module):
  '''Soft Dot Attention.

  Ref: http://www.aclweb.org/anthology/D15-1166
  Adapted from PyTorch OPEN NMT.
  '''
  
  def __init__(self, dim):
    '''Initialize layer.'''
    super(TextDotAttention, self).__init__()
    self.linear_in = nn.Linear(dim * 2, dim, bias=False)
    self.sm = nn.Softmax(dim=1)
    self.linear_out = nn.Linear(dim * 2, dim, bias=False)
    self.tanh = nn.Tanh()
  
  def forward(self, h, c, context, mask=None):
    '''Propagate h through the network.

    h: batch x dim
    context: batch x seq_len x dim
    mask: batch x seq_len indices to be masked
    '''
    target = self.linear_in(torch.cat((h, c), -1)).unsqueeze(
      2)  # batch x dim x 1
    
    # Get attention
    attn = torch.bmm(context, target).squeeze(2)  # batch x seq_len
    if mask is not None:
      # -Inf masking prior to the softmax
      attn.data.masked_fill_(mask, -float('inf'))
    attn = self.sm(attn)
    attn3 = attn.view(attn.size(0), 1, attn.size(1))  # batch x 1 x seq_len
    
    weighted_context = torch.bmm(attn3, context).squeeze(1)  # batch x dim
    h_tilde = torch.cat((weighted_context, h), 1)
    
    h_tilde = self.tanh(self.linear_out(h_tilde))
    return h_tilde, attn


class VisualSoftDotAttention(nn.Module):
  ''' Visual Dot Attention Layer. '''
  
  def __init__(self, h_dim, v_dim, dot_dim=256):
    '''Initialize layer.'''
    super(VisualSoftDotAttention, self).__init__()
    self.linear_in_h = nn.Linear(h_dim, dot_dim, bias=True)
    self.linear_in_v = nn.Linear(v_dim, dot_dim, bias=True)
    self.sm = nn.Softmax(dim=1)
  
  def forward(self, h, visual_context, mask=None):
    '''Propagate h through the network.

    h: batch x h_dim
    visual_context: batch x v_num x v_dim
    '''
    target = self.linear_in_h(h).unsqueeze(2)  # batch x dot_dim x 1
    context = self.linear_in_v(visual_context)  # batch x v_num x dot_dim
    
    # Get attention
    attn = torch.bmm(context, target).squeeze(2)  # batch x v_num
    attn = self.sm(attn)
    attn3 = attn.view(attn.size(0), 1, attn.size(1))  # batch x 1 x v_num
    
    weighted_context = torch.bmm(
      attn3, visual_context).squeeze(1)  # batch x v_dim
    return weighted_context, attn
