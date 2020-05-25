import torch
import config_lstur as conf

# refer to https://github.com/microsoft/recommenders/blob/master/reco_utils/recommender/newsrec/io/news_iterator.py
def parser_one_line(line, npratio=0, ID_spliter='%', col_spliter=' ', ):
    words = line.strip().split(ID_spliter)

    cols = words[0].strip().split(col_spliter)
    label = [float(i) for i in cols[: npratio + 1]]
    candidate_news_index = []
    click_news_index = []
    imp_index = []
    user_index = []

    for news in cols[npratio + 1 :]:
        tokens = news.split(":")
        if "Impression" in tokens[0]:
            imp_index.append(int(tokens[1]))
        elif "User" in tokens[0]:
            user_index.append(int(tokens[1]))
        elif "CandidateNews" in tokens[0]:
            # word index start by 0
            candidate_news_index.append([int(i) for i in tokens[1].split(",")])
        elif "ClickedNews" in tokens[0]:
            click_news_index.append([int(i) for i in tokens[1].split(",")])
        else:
            raise ValueError("data format is wrong")

    return label, imp_index, user_index, candidate_news_index, click_news_index

def load_all(test_data_file=False):
    max_user, max_item = 0, 0
    train_data = {}
    f = open(conf.train_data_path)
    for idx, line in enumerate(f):
        label, imp_index, user_index, candidate_news_index, click_news_index = parser_one_line(line, conf.npratio)
        train_data[idx] = [label, imp_index, user_index, candidate_news_index, click_news_index]

    val_data = {}
    f = open(conf.val_data_path)
    for idx, line in enumerate(f):
        label, imp_index, user_index, candidate_news_index, click_news_index = parser_one_line(line)
        val_data[idx] = [label, imp_index, user_index, candidate_news_index, click_news_index]
    
    if test_data_file:
        test_data = {}
        f = open(conf.test_data_path)
        for idx, line in enumerate(f):
            label, imp_index, user_index, candidate_news_index, click_news_index = parser_one_line(line)
            test_data[idx] = [label, imp_index, user_index, candidate_news_index, click_news_index]
    
        return train_data, val_data, test_data
    else:
        return train_data, val_data
        
class TrainData():
    def __init__(self, train_data):
        self.train_data = train_data
        self.length = len(train_data.keys())

    def _get_batch(self, batch_idx_list):
        user_indexes, his_input_title, pred_input_title, labels = [], [], [], []
        for batch_idx in batch_idx_list:
            user_indexes.append(self.train_data[batch_idx][2])
            labels.append(0)

            # get pred_input_title, 
            # [[b0_c0, b0_c1, ..., b0_cC], [b1_c0, b1_c1, ..., b1_cC], ...]
            for can_idx in range(conf.np_ratio+1):
                pred_input_title.append(self.train_data[batch_idx][3][can_idx])

        # get his_input_title, 
        # [[b0_h0, b1_h0, ..., bN_h0], [b0_h1, b0_h1, ..., bN_h1], ..., [b0_hm, b1_hm, ..., bN_hm], ..., [b0_hM, b1_hM, ..., bN_hM]]
        for his_idx in range(conf.his_size):
            for batch_idx in batch_idx_list:
                his_input_title.append(self.train_data[batch_idx][4][his_idx])

        return torch.LongTensor(user_indexes).cuda(),\
        torch.LongTensor(his_input_title).cuda(), \
        torch.LongTensor(pred_input_title).cuda(), \
        torch.LongTensor(labels).cuda()

class TestData():
    def __init__(self, test_data):
        self.test_data = test_data
        self.length = len(test_data.keys())

    def _get_batch(self, batch_idx_list):
        user_indexes, his_input_title, pred_input_title, labels = [], [], [], []
        imp_indexes = []
        for batch_idx in batch_idx_list:
            user_indexes.append(self.test_data[batch_idx][2])
            pred_input_title.append(self.test_data[batch_idx][3][0])

            labels.append(self.test_data[batch_idx][0][0])
            imp_indexes.append(self.test_data[batch_idx][1][0])

        # get his_input_title, 
        # [[b0_h0, b1_h0, ..., bN_h0], [b0_h1, b0_h1, ..., bN_h1], ..., [b0_hm, b1_hm, ..., bN_hm], ..., [b0_hM, b1_hM, ..., bN_hM]]
        for his_idx in range(conf.his_size):
            for batch_idx in batch_idx_list:
                his_input_title.append(self.test_data[batch_idx][4][his_idx])

        #import  pdb; pdb.set_trace()
        return torch.LongTensor(user_indexes).cuda(),\
        torch.LongTensor(his_input_title).cuda(), \
        torch.LongTensor(pred_input_title).cuda(), labels, imp_indexes