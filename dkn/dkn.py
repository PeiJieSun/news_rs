import torch
import torch.nn as nn
import torch.nn.functional as F

import config_dkn as conf 


class news_encoder(nn.Module):
    def __init__(self):
        super(news_encoder, self).__init__()

        self.word_embedding = nn.Embedding(conf.num_words, conf.word_dim)
        self.entity_embedding = nn.Embedding(conf.entity_size, conf.entity_dim)

        self.linear1 = nn.Linear(conf.entity_dim, conf.word_dim)
        
        self.conv2d = {}
        self.maxPool2d = {}
        for idx, k in enumerate(conf.kernel_list):
            self.conv2d[idx] = nn.Conv2d(in_channels=2, conf.num_filters, kernel_size=(k, conf.word_dim)))
            self.maxPool2d[idx] = nn.MaxPool2d((conf.doc_size-k+1, 1))

        self.reinit()

    def reinit(self):
        nn.init.zeros_(self.linear1.bias)
        nn.init.xavier_uniform_(self.linear1.weight)

        self.word_embedding.weight = nn.Parameter(torch.from_numpy(np.load(conf.word_embedding)).float())

        for idx, k in enumerate(conf.kernel_list):
            nn.init.zeros_(self.conv2d[idx].bias)
            nn.init.xavier_uniform_(self.conv2d[idx].weight)

    def forward(self, sequences_input_news):
        sequences_input_title, sequences_input_entity = sequences_input_news

        entity_emb = self.entity_embedding(sequences_input_entity) # (batch, doc_size, entity_dim)
        entity_emb = F.dropout(entity_emb, p=conf.dropout) # (batch, doc_size, entity_dim)
        g_e = torch.tanh(self.linear1(entity_emb)) # (batch, doc_size, word_dim)

        embedded_sequences_title = self.word_embedding(sequences_input_title) # (batch, doc_size, word_dim)
        y = F.dropout(embedded_sequences_title, p=conf.dropout) # (batch, doc_size, word_dim)

        W = torch.stack([y, g_e], dim=1) # (batch, 2, doc_size, word_dim)

        outs = []
        for idx, k in enumerate(conf.kernel_list):
            out = self.maxPool2d[idx](self.conv2d[idx](W)).view(-1, conf.num_filters) #(batch, num_filters)
            outs.append(out)    
        e_t = torch.cat(outs, dim=1) # (batch, 3*num_filters)

        e_t = F.dropout(e_t, p=conf.dropout) # (batch, 3*num_filters)
        
        return e_t

class user_encoder(nn.Module):
    def __init__(self, news_encoder):
        super(user_encoder, self).__init__()
        self.news_encoder = news_encoder

        self.multiheadAttention = nn.MultiheadAttention(conf.num_filters, 1)
    
    # news_present: (batch, candidate, num_filters)
    def forward(self, his_input_news, news_present):        
        e_t = self.news_encoder(his_input_news).view(-1, conf.his_size, conf.num_filters) # (batch, his_size, num_filters)
        e_t = e_t.transpose(0, 1)

        user_present = self.multiheadAttention(news_present.transpose(0, 1), \
            e_t, e_t, need_weights=False) # (candidate, batch, num_filters)

        user_present = user_present.transpose(0, 1) # (batch, candidate, num_filters)
        return user_present

class dkn(nn.Module): 
    def __init__(self):
        super(dkn, self).__init__()
        
        self.news_encoder = news_encoder()
        self.user_encoder = user_encoder(self.news_encoder)
        
    def forward(self, his_input_news, pred_input_news, labels):
        news_present = self.news_encoder(pred_input_news).view(-1, conf.npratio+1, conf.num_filters) # (batch, candidate, num_filters)
        user_present = self.user_encoder(his_input_news, news_present) # (batch, candidate, num_filters)
        
        preds = torch.matmul(news_present, user_present).view(-1, conf.npratio+1) # (batch, candidate)
        obj = F.cross_entropy(preds, labels, reduction='mean')

        return obj

    def predict(self, his_input_title, pred_input_title):
        news_present = self.news_encoder(pred_input_title) # (batch, num_filters)
        user_present = self.user_encoder(his_input_title, news_present.view(-1, 1, conf.num_filters)) # (batch, 1, num_filters)
        user_present = user_present.view(-1, conf.num_filters) # (batch, num_filters)

        preds = torch.sigmoid(torch.sum(news_present * user_present, dim=1, keepdims=True))
        return preds.view(-1)