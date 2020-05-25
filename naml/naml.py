import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

import config_naml as conf 

class title_encoder(nn.Module):
    def __init__(self, word_embedding):
        super(title_encoder, self).__init__()

        self.word_embedding = word_embedding
        self.Conv1d = nn.Conv1d(conf.word_dim, conf.num_filters, conf.kernel_size, padding=1)
        self.linear_1 = nn.Linear(conf.num_filters, 1)
        
        self.reinit()

    def reinit(self):
        nn.init.zeros_(self.linear_1.bias)
        nn.init.xavier_uniform_(self.linear_1.weight)

        nn.init.zeros_(self.Conv1d.bias)
        nn.init.xavier_uniform_(self.Conv1d.weight)

    def forward(self, sequences_input_title):
        # sequences_input_title: (batch, title_length)
        embedded_sequences_title = self.word_embedding(sequences_input_title) # (batch, title_length, word_dimension)
        y = F.dropout(embedded_sequences_title, p=conf.dropout) # (batch, title_length, word_dimension)
        y = y.transpose(1, 2) # (batch, word_dimension, title_length)
        y = self.Conv1d(y) # (batch, num_filters, title_length)
        y = y.transpose(1, 2) # (batch, title_length, num_filters)
        y = F.dropout(y, p=conf.dropout) # (batch, title_length, num_filters)

        mask = (sequences_input_title>0).long() # (batch, title_length)

        a_i = torch.tanh(self.linear_1(y)).view(-1, sequences_input_title.shape[1]) # (batch, title_length)
        a_i = torch.exp(a_i) # (batch, title_length)
        a_i = torch.mul(a_i, mask) # (batch, title_length)
        b_i = torch.sum(a_i, 1, keepdims=True) + 1e-6 # (batch, 1)
        alpha = torch.div(a_i, b_i).view(-1, sequences_input_title.shape[1], 1) # (batch, title_length, 1)

        e_t = torch.sum(torch.mul(alpha, y), 1) # (batch, num_filters)

        return e_t

class body_encoder(nn.Module):
    def __init__(self, word_embedding):
        super(body_encoder, self).__init__()

        self.word_embedding = word_embedding
        self.Conv1d = nn.Conv1d(conf.word_dim, conf.num_filters, conf.kernel_size, padding=1)
        self.linear_1 = nn.Linear(conf.num_filters, 1)
        
        self.reinit()

    def reinit(self):
        nn.init.zeros_(self.linear_1.bias)
        nn.init.xavier_uniform_(self.linear_1.weight)

        nn.init.zeros_(self.Conv1d.bias)
        nn.init.xavier_uniform_(self.Conv1d.weight)

    def forward(self, sequences_input_body):
        # sequences_input_body: (batch, body_length)
        embedded_sequences_title = self.word_embedding(sequences_input_body) # (batch, body_length, word_dimension)
        y = F.dropout(embedded_sequences_title, p=conf.dropout) # (batch, body_length, word_dimension)
        y = y.transpose(1, 2) # (batch, word_dimension, body_length)
        y = self.Conv1d(y) # (batch, num_filters, body_length)
        y = y.transpose(1, 2) # (batch, body_length, num_filters)
        y = F.dropout(y, p=conf.dropout) # (batch, body_length, num_filters)

        mask = (sequences_input_body>0).long() # (batch, body_length)

        a_i = torch.tanh(self.linear_1(y)).view(-1, sequences_input_body.shape[1]) # (batch, body_length)
        a_i = torch.exp(a_i) # (batch, body_length)
        a_i = torch.mul(a_i, mask) # (batch, body_length)
        b_i = torch.sum(a_i, 1, keepdims=True) + 1e-6 # (batch, 1)
        alpha = torch.div(a_i, b_i).view(-1, sequences_input_body.shape[1], 1) # (batch, body_length, 1)

        e_t = torch.sum(torch.mul(alpha, y), 1) # (batch, num_filters)

        return e_t

class vert_encoder(nn.Module):
    def __init__(self):
        super(vert_encoder, self).__init__()

        self.vert_embedding = nn.Embedding(conf.vert_num, conf.vert_emb_dim)
        self.linear_1 = nn.Linear(conf.vert_emb_dim, conf.num_filters)

        self.reinit()

    def reinit(self):
        nn.init.zeros_(self.linear_1.bias)
        nn.init.xavier_uniform_(self.linear_1.weight)

    def forward(self, input_vert):
        vert_emb = self.vert_embedding(input_vert)
        pred_vert = self.linear_1(vert_emb).view(-1, conf.num_filters)

        return pred_vert

class subvert_encoder(nn.Module):
    def __init__(self):
        super(subvert_encoder, self).__init__()

        self.subvert_embedding = nn.Embedding(conf.subvert_num, conf.subvert_emb_dim)
        self.linear_1 = nn.Linear(conf.subvert_emb_dim, conf.num_filters)

        self.reinit()

    def reinit(self):
        nn.init.zeros_(self.linear_1.bias)
        nn.init.xavier_uniform_(self.linear_1.weight)

    def forward(self, input_subvert):
        subvert_emb = self.subvert_embedding(input_subvert)
        pred_subvert = self.linear_1(subvert_emb).view(-1, conf.num_filters)

        return pred_subvert

class news_encoder(nn.Module):
    def __init__(self):
        super(news_encoder, self).__init__()

        self.word_embedding = nn.Embedding(conf.num_words, conf.word_dim)

        self.title_encoder = title_encoder(self.word_embedding)
        self.body_encoder = body_encoder(self.word_embedding)
        self.vert_encoder = vert_encoder()
        self.subvert_encoder = subvert_encoder()
        
        self.linear_1 = nn.Linear(conf.num_filters, 1)

        self.reinit()

    def reinit(self):
        self.word_embedding.weight = nn.Parameter(torch.from_numpy(np.load(conf.word_embedding)).float())

        nn.init.zeros_(self.linear_1.bias)
        nn.init.xavier_uniform_(self.linear_1.weight)

    def forward(self, input_news):
        sequences_input_title, sequences_input_body, input_vert, input_subvert = input_news

        title_repr = self.title_encoder(sequences_input_title) # (batch, num_filters)
        body_repr = self.body_encoder(sequences_input_body) # (batch, num_filters)
        vert_repr = self.vert_encoder(input_vert) # (batch, num_filters)
        subvert_repr = self.subvert_encoder(input_subvert) # (batch, num_filters)
        
        news_repr = [title_repr, body_repr, vert_repr, subvert_repr]
        news_repr = torch.stack(news_repr, dim=0) # (4, batch, num_filters)
        news_repr = news_repr.transpose(0, 1) # (batch, 4, num_filters)
        news_repr = news_repr.reshape(-1, conf.num_filters) # (batch*4, num_filters)

        # attention
        a_i = torch.tanh(self.linear_1(news_repr)).view(-1, 4) # (batch, 4)
        a_i = torch.exp(a_i) # (batch, 4)
        b_i = torch.sum(a_i, 1, keepdims=True) + 1e-6 # (batch, 1)
        alpha = torch.div(a_i, b_i).view(-1, 4, 1) # (batch, 4, 1)

        news_repr = news_repr.reshape(-1, 4, conf.num_filters) # (batch, 4, num_filters)
        e_t = torch.sum(torch.mul(alpha, news_repr), 1) # (batch, num_filters)

        return e_t

class user_encoder(nn.Module):
    def __init__(self, news_encoder):
        super(user_encoder, self).__init__()
        self.news_encoder = news_encoder
        
        self.linear_1 = nn.Linear(conf.num_filters, 1)

        self.reinit()

    def reinit(self):
        nn.init.zeros_(self.linear_1.bias)
        nn.init.xavier_uniform_(self.linear_1.weight)

    def forward(self, his_input_news):
        e_t = self.news_encoder(his_input_news) # (batch*his_size, num_filters)

        # attention
        a_i = torch.tanh(self.linear_1(e_t)).view(-1, conf.his_size) # (batch, his_size)
        a_i = torch.exp(a_i) # (batch, his_size)
        b_i = torch.sum(a_i, 1, keepdims=True) + 1e-6 # (batch, 1)
        alpha = torch.div(a_i, b_i).view(-1, conf.his_size, 1) # (batch, 4, 1)

        e_t = e_t.reshape(-1, conf.his_size, conf.num_filters) # (batch, his_size, num_filters)
        user_present = torch.sum(torch.mul(alpha, e_t), 1) # (batch, num_filters)

        return user_present

class naml(nn.Module):
    def __init__(self):
        super(naml, self).__init__()
        
        self.news_encoder = news_encoder()
        self.user_encoder = user_encoder(self.news_encoder)
        
    def forward(self, his_input_news, pred_input_news, labels):
        news_present = self.news_encoder(pred_input_news).view(-1, conf.npratio+1, conf.num_filters) # (batch, candidate, num_filters)
        user_present = self.user_encoder(his_input_news).view(-1, conf.num_filters, 1) # (batch, gru_unit, 1)

        preds = torch.matmul(news_present, user_present).view(-1, conf.npratio+1) # (batch, candidate)
        obj = F.cross_entropy(preds, labels, reduction='mean')

        return obj

    def predict(self, his_input_news, pred_input_news):
        news_present = self.news_encoder(pred_input_news) # (batch, num_filters)
        user_present = self.user_encoder(his_input_news) # (batch, num_filters)

        preds = torch.sigmoid(torch.sum(news_present * user_present, dim=1, keepdims=True))
        return preds.view(-1)