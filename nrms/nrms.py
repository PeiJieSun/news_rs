import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

import config_nrms as conf 

class news_encoder(nn.Module):
    def __init__(self):
        super(news_encoder, self).__init__()
        self.word_embedding = nn.Embedding(conf.num_words, conf.word_dim)
        
        self.multi_head_attention = nn.MultiheadAttention(conf.word_dim, 4)
        self.linear_1 = nn.Linear(conf.word_dim, 1)

        self.reinit()

    def reinit(self):
        nn.init.zeros_(self.linear_1.bias)
        nn.init.xavier_uniform_(self.linear_1.weight)

        self.word_embedding.weight = \
            nn.Parameter(torch.from_numpy(np.load(conf.word_embedding)).float())

    def forward(self, sequences_input_title):
        # sequences_input_title: (batch, sequence_length)
        embedded_sequences_title = self.word_embedding(sequences_input_title) # (batch, sequence_length, word_dimension)
        y = F.dropout(embedded_sequences_title, p=conf.dropout) # (batch, sequence_length, word_dimension)
        y = y.transpose(0, 1) # (sequence_length, batch, word_dimension)

        attn_output, attn_output_weights = self.multi_head_attention(y, y, y)
        h_out = F.dropout(attn_output, p=conf.dropout) # (sequence_length, batch, word_dimension)
        h_out = h_out.transpose(0, 1) # (batch, sequence_length, word_dim)

        mask = (sequences_input_title>0).long() # (batch, sequence_length)

        a_i = torch.tanh(self.linear_1(h_out)).view(-1, sequences_input_title.shape[1]) # (batch, sequence_length)
        a_i = torch.exp(a_i) # (batch, sequence_length)
        a_i = torch.mul(a_i, mask) # (batch, sequence_length)
        b_i = torch.sum(a_i, 1, keepdims=True) + 1e-6 # (batch, 1)
        alpha = torch.div(a_i, b_i).view(-1, sequences_input_title.shape[1], 1) # (batch, sequence_length, 1)

        e_t = torch.sum(torch.mul(alpha, h_out), 1) # (batch, num_filters)

        return e_t

class user_encoder(nn.Module):
    def __init__(self, news_encoder):
        super(user_encoder, self).__init__()
        self.news_encoder = news_encoder

        self.multi_head_attention = nn.MultiheadAttention(conf.word_dim, 4)
        self.linear_1 = nn.Linear(conf.word_dim, 1)

        self.reinit()

    def reinit(self):
        nn.init.zeros_(self.linear_1.bias)
        nn.init.xavier_uniform_(self.linear_1.weight)

    def forward(self, his_input_title):
        mask_his_input_title = his_input_title.view(-1, conf.his_size, conf.doc_size) # (batch, his_size, sequence_length)
        mask_his_input_title = torch.sum(mask_his_input_title, dim=2) # (batch, his_length)

        e_t = self.news_encoder(his_input_title).view(
            conf.his_size, -1, conf.word_dim
        ) # (batch, his_length, word_dim)

        attn_output, attn_output_weights = self.multi_head_attention(e_t, e_t, e_t)
        h_out = attn_output.transpose(0, 1) # (batch, his_length, word_dim)
        
        mask = (mask_his_input_title>0).long() # (batch, sequence_length)

        a_i = torch.tanh(self.linear_1(h_out)).view(-1, mask_his_input_title.shape[1]) # (batch, sequence_length)
        a_i = torch.exp(a_i) # (batch, sequence_length)
        a_i = torch.mul(a_i, mask) # (batch, sequence_length)
        b_i = torch.sum(a_i, 1, keepdims=True) + 1e-6 # (batch, 1)
        alpha = torch.div(a_i, b_i).view(-1, mask_his_input_title.shape[1], 1) # (batch, sequence_length, 1)

        user_present = torch.sum(torch.mul(alpha, h_out), 1) # (batch, num_filters)
 
        return user_present

class nrms(nn.Module): 
    def __init__(self):
        super(nrms, self).__init__()
        
        self.news_encoder = news_encoder()
        self.user_encoder = user_encoder(self.news_encoder)

    # his_input_title: (batch*his_size, sequence_length)
    # pred_input_title: (batch*candidate, sequence_length)
    def forward(self, his_input_title, pred_input_title, labels):
        news_present = self.news_encoder(pred_input_title).view(\
            -1, conf.npratio+1, conf.word_dim) # (batch, candidate, word_dim)
        user_present = self.user_encoder(his_input_title).view(\
            -1, conf.word_dim, 1) # (batch, word_dim, 1)

        preds = torch.matmul(news_present, user_present).view(-1, conf.npratio+1) # (batch, candidate)
        obj = F.cross_entropy(preds, labels, reduction='mean')

        return obj

    def predict(self, his_input_title, pred_input_title):
        news_present = self.news_encoder(pred_input_title) 
        user_present = self.user_encoder(his_input_title) 

        preds = torch.sigmoid(torch.sum(news_present * user_present, dim=1, keepdims=True))
        return preds.view(-1)