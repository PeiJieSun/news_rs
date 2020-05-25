import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

import config_nrms as conf 

class multi_head_attention(nn.Module):
    def __init__(self, head_dim, num_heads, total_dim):
        super(multi_head_attention, self).__init__()

        self.Q = torch.randn(num_heads, total_dim, total_dim, requires_grad=True).cuda()
        self.V = torch.randn(num_heads, total_dim, conf.head_dim, requires_grad=True).cuda()

        self.reinit()
    
    def reinit(self):
        nn.init.xavier_uniform_(self.Q)
        nn.init.xavier_uniform_(self.V)

    def self_attention(self, y, sequences_input_title, k):
        a1 = torch.matmul(y, self.Q[k]) # (batch, sequence_length, word_dimension)
        a2 = torch.matmul(a1, y.transpose(1, 2)) # (batch, sequence_length, sequence_length)
        a3 = torch.exp(a2) # (batch, sequence_length, sequence_length)
        mask = (sequences_input_title>0).long().view(-1, 1, sequences_input_title.shape[1]) # (batch, 1, sequence_length)
        a4 = torch.mul(a3, mask) # (batch, sequence_length, sequence_length)
        a5 = torch.sum(a4, dim=-1, keepdims=True) + 1e-6 # (batch, sequence_length, 1)
        alpha = torch.div(a3, a5) # (batch, sequence_length, sequence_length)

        h0 = torch.matmul(alpha, y) # (batch, sequence_length, word_dimension)
        h1 = torch.matmul(h0, self.V[k]) # (batch, sequence_length, head_dim)

        return h1

    def forward(self, y, sequences_input_title):
        h_out = []
        for k in range(conf.num_heads):
            h1 = self.self_attention(y, sequences_input_title, k)
            h_out.append(h1)
        h_out = torch.cat(h_out, dim=-1) # (batch, sequence_length, num_heads*head_num)
        return h_out

class news_encoder(nn.Module):
    def __init__(self):
        super(news_encoder, self).__init__()
        self.word_embedding = nn.Embedding(conf.num_words, conf.word_dim)
        
        self.multi_head_attention = \
            multi_head_attention(conf.head_dim, conf.num_heads, conf.word_dim)
        self.linear_1 = nn.Linear(conf.head_dim*conf.num_heads, 200)
        self.linear_2 = nn.Linear(200, 1, bias=False)

        self.reinit()

    def reinit(self):
        nn.init.zeros_(self.linear_1.bias)
        nn.init.xavier_uniform_(self.linear_1.weight)

        nn.init.xavier_uniform_(self.linear_2.weight)

        self.word_embedding.weight = \
            nn.Parameter(torch.from_numpy(np.load(conf.word_embedding)).float())

    def forward(self, sequences_input_title):
        # sequences_input_title: (batch, sequence_length)
        embedded_sequences_title = self.word_embedding(sequences_input_title) # (batch, sequence_length, word_dimension)
        y = F.dropout(embedded_sequences_title, p=conf.dropout)
        h_out = self.multi_head_attention(y, sequences_input_title) # (batch, sequence_length, num_heads*head_dim)
        h_out = F.dropout(h_out, p=conf.dropout)

        mask = (sequences_input_title>0).long() # (batch, sequence_length)

        a_i = torch.tanh(self.linear_1(h_out))#.view(-1, sequences_input_title.shape[1]) # (batch, sequence_length)
        a_i = self.linear_2(a_i).view(-1, sequences_input_title.shape[1])
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

        self.multi_head_attention = \
            multi_head_attention(conf.head_dim, conf.num_heads, conf.head_dim*conf.num_heads)

        self.linear_1 = nn.Linear(conf.head_dim*conf.num_heads, 200)
        self.linear_2 = nn.Linear(200, 1, bias=False)

        self.reinit()

    def reinit(self):
        nn.init.zeros_(self.linear_1.bias)
        nn.init.xavier_uniform_(self.linear_1.weight)

        nn.init.xavier_uniform_(self.linear_2.weight)

    def forward(self, his_input_title):
        mask_his_input_title = his_input_title.view(-1, conf.his_size, conf.doc_size) # (batch, his_size, sequence_length)
        mask_his_input_title = torch.sum(mask_his_input_title, dim=2) # (batch, his_length)

        e_t = self.news_encoder(his_input_title).view(
            -1, conf.his_size, conf.num_heads*conf.head_dim
        ) # (batch, his_length, num_heads*head_dim)

        h_out = self.multi_head_attention(e_t, mask_his_input_title) # (batch, his_length, num_heads*head_dim)
        
        mask = (mask_his_input_title>0).long() # (batch, sequence_length)

        a_i = torch.tanh(self.linear_1(h_out))#.view(-1, mask_his_input_title.shape[1]) # (batch, sequence_length)
        a_i = self.linear_2(a_i).view(-1, mask_his_input_title.shape[1])
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

    #his_input_title: (batch*his_size, sequence_length)
    #pred_input_title: (batch*candidate, sequence_length)
    def forward(self, his_input_title, pred_input_title, labels):
        news_present = self.news_encoder(pred_input_title).view(\
            -1, conf.npratio+1, conf.head_dim*conf.num_heads) # (batch, candidate, num_heads*head_dim)
        user_present = self.user_encoder(his_input_title).view(\
            -1, conf.head_dim*conf.num_heads, 1) # (batch, num_heads*head_dim, 1)

        preds = torch.matmul(news_present, user_present).view(-1, conf.npratio+1) # (batch, candidate)
        obj = F.cross_entropy(preds, labels, reduction='mean')

        return obj

    def predict(self, his_input_title, pred_input_title):
        news_present = self.news_encoder(pred_input_title) 
        user_present = self.user_encoder(his_input_title) 

        preds = torch.sigmoid(torch.sum(news_present * user_present, dim=1, keepdims=True))
        return preds.view(-1)