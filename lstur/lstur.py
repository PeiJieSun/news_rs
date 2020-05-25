import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import config_lstur as conf 

class news_encoder(nn.Module):
    def __init__(self):
        super(news_encoder, self).__init__()

        self.word_embedding = nn.Embedding(conf.num_words, conf.word_dim)
        self.Conv1d = nn.Conv1d(conf.word_dim, conf.num_filters, conf.kernel_size, padding=1)
        self.linear_1 = nn.Linear(conf.num_filters, 1)
        
        self.reinit()

    def reinit(self):
        nn.init.zeros_(self.linear_1.bias)
        nn.init.xavier_uniform_(self.linear_1.weight)

        nn.init.zeros_(self.Conv1d.bias)
        nn.init.xavier_uniform_(self.Conv1d.weight)

        self.word_embedding.weight = nn.Parameter(torch.from_numpy(np.load(conf.word_embedding)).float())

    def forward(self, sequences_input_title):
        # sequences_input_title: (batch, sequence_length)
        embedded_sequences_title = self.word_embedding(sequences_input_title) # (batch, sequence_length, word_dimension)
        y = F.dropout(embedded_sequences_title, p=conf.dropout) # (batch, sequence_length, word_dimension)
        y = y.transpose(1, 2) # (batch, word_dimension, sequence_length)
        y = self.Conv1d(y) # (batch, num_filters, sequence_length)
        y = y.transpose(1, 2) # (batch, sequence_length, num_filters)
        y = F.dropout(y, p=conf.dropout) # (batch, sequence_length, num_filters)

        mask = (sequences_input_title>0).long() # (batch, sequence_length)

        a_i = torch.tanh(self.linear_1(y)).view(-1, conf.doc_size) # (batch, sequence_length)
        a_i = torch.exp(a_i) # (batch, sequence_length)
        a_i = torch.mul(a_i, mask) # (batch, sequence_length)
        b_i = torch.sum(a_i, 1, keepdims=True) + 1e-6 # (batch, 1)
        alpha = torch.div(a_i, b_i).view(-1, conf.doc_size, 1) # (batch, sequence_length, 1)

        e_t = torch.sum(torch.mul(alpha, y), 1) # (batch, num_filters)

        return e_t

class user_encoder(nn.Module):
    def __init__(self, news_encoder):
        super(user_encoder, self).__init__()

        self.news_encoder = news_encoder
        self.user_embedding = nn.Embedding(conf.num_users, conf.gru_unit)
        self.user_bias = nn.Embedding(conf.num_users, 1)
        self.rnn = nn.GRU(conf.num_filters, conf.gru_unit)

        self.linear_1 = nn.Linear(2*conf.gru_unit, conf.gru_unit)

        self.reinit()

    def reinit(self):
        nn.init.zeros_(self.linear_1.bias)
        nn.init.xavier_uniform_(self.linear_1.weight)

        nn.init.zeros_(self.user_bias.weight)
        nn.init.zeros_(self.user_embedding.weight)

        for name, param in self.rnn.named_parameters():
            if 'bias' in name:
                nn.init.zeros_(param)
            elif 'weight' in name:
                nn.init.xavier_uniform_(param)

    def forward(self, user_indexes, his_input_title, type='ini'):
        long_u_embed = self.user_embedding(user_indexes) + self.user_bias(user_indexes) # (batch, gru_unit)

        e_t = self.news_encoder(his_input_title).view(conf.his_size, -1, conf.num_filters) # (his_length*batch, num_filters)

        if type == 'ini':
            hidden_state = long_u_embed.view(1, -1, conf.gru_unit)
            outputs, _ = self.rnn(e_t, hidden_state)
            user_present = outputs[-1]
        elif type == 'con':
            outputs, _ = self.rnn(e_t)
            user_present = torch.cat(long_u_embed, outputs[-1])
            user_present = self.linear_1(user_present)

        return user_present

class lstur(nn.Module): 
    def __init__(self):
        super(lstur, self).__init__()
        
        self.news_encoder = news_encoder()
        self.user_encoder = user_encoder(self.news_encoder)
        
    def forward(self, user_indexes, his_input_title, pred_input_title, labels):
        news_present = self.news_encoder(pred_input_title).view(-1, conf.np_ratio+1, conf.num_filters) # (batch, candidate, num_filters)
        user_present = self.user_encoder(user_indexes, his_input_title).view(-1, conf.gru_unit, 1) # (batch, gru_unit, 1)

        preds = torch.matmul(news_present, user_present).view(-1, conf.np_ratio+1) # (batch, candidate)
        #import pdb; pdb.set_trace()
        obj = F.cross_entropy(preds, labels, reduction='mean')

        return obj

    def predict(self, user_indexes, his_input_title, pred_input_title):
        news_present = self.news_encoder(pred_input_title) # (batch, num_filters)
        user_present = self.user_encoder(user_indexes, his_input_title) # (batch, gru_unit)

        preds = torch.sigmoid(torch.sum(news_present * user_present, dim=1, keepdims=True))
        return preds.view(-1)