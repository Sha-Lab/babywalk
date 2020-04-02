import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from cuda import try_cuda
from attentions import SoftDotAttention, VisualSoftDotAttention
from context_encoder import ContextEncoder


class EltwiseProdScoring(nn.Module):
    '''
    Linearly mapping h and v to the same dimension, and do a elementwise
    multiplication and a linear scoring
    '''
    
    def __init__(self, h_dim, a_dim, dot_dim=256):
        '''Initialize layer.'''
        super(EltwiseProdScoring, self).__init__()
        self.linear_in_h = nn.Linear(h_dim, dot_dim, bias=True)
        self.linear_in_a = nn.Linear(a_dim, dot_dim, bias=True)
        self.linear_out = nn.Linear(dot_dim, 1, bias=True)
    
    def forward(self, h, all_u_t, mask=None):
        '''Propagate h through the network.

        h: batch x h_dim
        all_u_t: batch x a_num x a_dim
        '''
        target = self.linear_in_h(h).unsqueeze(1)  # batch x 1 x dot_dim
        context = self.linear_in_a(all_u_t)  # batch x a_num x dot_dim
        eltprod = torch.mul(target, context)  # batch x a_num x dot_dim
        logits = self.linear_out(eltprod).squeeze(2)  # batch x a_num
        return logits


class EltwiseProdScoringWithContext(nn.Module):
    '''
    Linearly mapping h and v to the same dimension, and do a elementwise
    multiplication and a linear scoring
    '''
    
    def __init__(self, h_dim, a_dim, dot_dim=512, dropout=0.5):
        '''Initialize layer.'''
        super(EltwiseProdScoringWithContext, self).__init__()
        self.linear_combine = nn.Sequential(
            nn.Linear(h_dim * 3, dot_dim, bias=True),
            nn.ReLU(),
            nn.Linear(dot_dim, dot_dim, bias=True)
        )
        self.linear_in_a = nn.Linear(a_dim, dot_dim, bias=True)
        self.linear_out = nn.Linear(dot_dim, 1, bias=True)
    
    def forward(self, h, context, text_context, all_u_t, mask=None):
        '''Propagate h through the network.

        h: batch x h_dim
        all_u_t: batch x a_num x a_dim
        '''
        combine = torch.cat([F.normalize(h), F.normalize(context), F.normalize(text_context)], dim=1)
        target = self.linear_combine(combine).unsqueeze(1)  # batch x 1 x dot_dim
        actions = self.linear_in_a(all_u_t)  # batch x a_num x dot_dim
        eltprod = torch.mul(target, actions)  # batch x a_num x dot_dim
        logits = self.linear_out(eltprod).squeeze(2)  # batch x a_num
        return logits


class EncoderLSTM(nn.Module):
    ''' Encodes navigation instructions, returning hidden state context (for
        attention methods) and a decoder initial state. '''
    
    def __init__(self, vocab_size, embedding_size, hidden_size, padding_idx, dropout_ratio,
                 bidirectional=False, num_layers=1, glove=None, subout='tanh'):
        super(EncoderLSTM, self).__init__()
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.drop = nn.Dropout(p=dropout_ratio)
        self.num_directions = 2 if bidirectional else 1
        self.num_layers = num_layers
        self.embedding = nn.Embedding(vocab_size, embedding_size, padding_idx)
        self.subout = subout
        self.use_glove = glove is not None
        if self.use_glove:
            print('Using GloVe embedding')
            self.embedding.weight.data[...] = torch.from_numpy(glove)
            self.embedding.weight.requires_grad = False
        self.lstm = nn.LSTM(embedding_size, hidden_size, self.num_layers,
                            batch_first=True, dropout=dropout_ratio if self.num_layers > 1 else 0,
                            bidirectional=bidirectional)
        self.encoder2decoder = nn.Linear(hidden_size * self.num_directions,
                                         hidden_size * self.num_directions)
    
    def load_my_state_dict(self, state_dict):
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name not in own_state:
                continue
            if isinstance(param, nn.Parameter):
                param = param.data
            own_state[name].copy_(param)
    
    def init_state(self, batch_size):
        ''' Initialize to zero cell states and hidden states.'''
        h0 = Variable(torch.zeros(
            self.num_layers * self.num_directions,
            batch_size,
            self.hidden_size
        ), requires_grad=False)
        c0 = Variable(torch.zeros(
            self.num_layers * self.num_directions,
            batch_size,
            self.hidden_size
        ), requires_grad=False)
        return try_cuda(h0), try_cuda(c0)
    
    def forward(self, *args, **kwargs):
        if 'context' in kwargs and kwargs['context'] == True:
            return self.forward_context(*args)
        else:
            return self.forward_current(*args)
    
    def forward_current(self, inputs, lengths, max_length):
        ''' Expects input vocab indices as (batch, seq_len). Also requires a
            list of lengths for dynamic batching. '''
        batch_size = inputs.size(0)
        embeds = self.embedding(inputs)  # (batch, seq_len, embedding_size)
        if not self.use_glove:
            embeds = self.drop(embeds)
        h0, c0 = self.init_state(batch_size)
        packed_embeds = pack_padded_sequence(embeds, lengths, enforce_sorted=False, batch_first=True)
        # self.lstm.flatten_parameters()
        enc_h, (enc_h_t, enc_c_t) = self.lstm(packed_embeds, (h0, c0))
        
        if self.num_directions == 2:
            h_t = torch.cat((enc_h_t[-1], enc_h_t[-2]), 1)
            c_t = torch.cat((enc_c_t[-1], enc_c_t[-2]), 1)
        else:
            h_t = enc_h_t[-1]
            c_t = enc_c_t[-1]  # (batch, hidden_size)
        
        ctx, lengths = pad_packed_sequence(enc_h, batch_first=True)
        
        if self.subout == "tanh":
            decoder_init = nn.Tanh()(self.encoder2decoder(h_t))
        elif self.subout == "max":
            ctx_max, _ = ctx.max(1)
            decoder_init = nn.Tanh()(self.encoder2decoder(ctx_max))
        else:
            assert False, "Subout not available!"
        
        out_dims = (batch_size, max_length, self.hidden_size)
        out_ctx = ctx.data.new(*out_dims).fill_(0)
        out_ctx[:, :max(lengths), :] = self.drop(ctx)
        return out_ctx, decoder_init, c_t
    
    def forward_context(self, inputs, lengths):
        batch_size = inputs.size(0)
        embeds = self.embedding(inputs)  # (batch, seq_len, embedding_size)
        if not self.use_glove:
            embeds = self.drop(embeds)
        h0, c0 = self.init_state(batch_size)
        packed_embeds = pack_padded_sequence(embeds, lengths, enforce_sorted=False, batch_first=True)
        # self.lstm.flatten_parameters()
        enc_h, (enc_h_t, enc_c_t) = self.lstm(packed_embeds, (h0, c0))
        
        if self.num_directions == 2:
            h_t = torch.cat((enc_h_t[-1], enc_h_t[-2]), 1)
        else:
            h_t = enc_h_t[-1]
        return h_t


class AttnDecoderLSTM(nn.Module):
    '''
    An unrolled LSTM with attention over instructions for decoding navigation
    actions.
    '''
    
    def __init__(self, embedding_size, hidden_size, dropout_ratio, feature_size, history=False):
        super(AttnDecoderLSTM, self).__init__()
        self.embedding_size = embedding_size
        self.feature_size = feature_size
        self.hidden_size = hidden_size
        self.drop = nn.Dropout(p=dropout_ratio)
        self.lstm = nn.LSTMCell(embedding_size + feature_size, hidden_size)
        self.visual_attention_layer = VisualSoftDotAttention(hidden_size, feature_size)
        self.text_attention_layer = SoftDotAttention(hidden_size)
        if history:
            self.linear_context_out = nn.Linear(hidden_size, hidden_size, bias=True)
            self.linear_text_out = nn.Linear(hidden_size, hidden_size, bias=True)
            self.context_encoder = ContextEncoder(feature_size, hidden_size, dropout_ratio, True)
            self.decoder2action_text_context = EltwiseProdScoringWithContext(hidden_size, embedding_size)
        else:
            self.decoder2action = EltwiseProdScoring(hidden_size, embedding_size)
    
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
            return self.forward_current(*args, ctx_mask=kwargs['ctx_mask'], history_context=kwargs['history_context'])
    
    def forward_current(self, u_t_prev, all_u_t, visual_context, h_0, c_0, ctx, ctx_mask=None, history_context=None):
        ''' Takes a single step in the decoder LSTM (allowing sampling).

        u_t_prev: batch x embedding_size
        all_u_t: batch x a_num x embedding_size
        visual_context: batch x v_num x feature_size
        h_0: batch x hidden_size
        c_0: batch x hidden_size
        ctx: batch x seq_len x dim
        ctx_mask: batch x seq_len - indices to be masked
        '''
        feature, alpha_v = self.visual_attention_layer(h_0, visual_context)
        concat_input = torch.cat((u_t_prev, feature), 1)
        concat_drop = self.drop(concat_input)
        
        h_1, c_1 = self.lstm(concat_drop, (h_0, c_0))
        h_1_drop = self.drop(h_1)
        h_tilde, alpha = self.text_attention_layer(h_1_drop, ctx, ctx_mask)
        
        if history_context is not None:
            context = self.linear_context_out(history_context[0])
            text = self.linear_text_out(history_context[1])
            logit = self.decoder2action_text_context(h_tilde, context, text, all_u_t)
        else:
            logit = self.decoder2action(h_tilde, all_u_t)
        return h_1, c_1, alpha, logit, alpha_v
    
    def forward_context(self, *args):
        return self.context_encoder(*args)
