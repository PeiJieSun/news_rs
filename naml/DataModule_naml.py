import torch
import config_naml as conf

# refer to https://github.com/microsoft/recommenders/blob/master/reco_utils/recommender/newsrec/io/naml_iterator.py
def parser_one_line(line, npratio=0, ID_spliter='%', col_spliter=' ', ):
    words = line.strip().split(ID_spliter)

    cols = words[0].strip().split(col_spliter)
    label = [float(i) for i in cols[: npratio + 1]]
    candidate_title_index = []
    click_title_index = []
    candidate_body_index = []
    click_body_index = []
    candidate_vert_index = []
    click_vert_index = []
    candidate_subvert_index = []
    click_subvert_index = []
    imp_index = []
    user_index = []

    for news in cols[npratio + 1 :]:
        tokens = news.split(":")
        if "Impression" in tokens[0]:
            imp_index.append(int(tokens[1]))
        elif "User" in tokens[0]:
            user_index.append(int(tokens[1]))
        elif "CandidateTitle" in tokens[0]:
            # word index start by 0
            candidate_title_index.append([int(i) for i in tokens[1].split(",")])
        elif "ClickedTitle" in tokens[0]:
            click_title_index.append([int(i) for i in tokens[1].split(",")])
        elif "CandidateBody" in tokens[0]:
            candidate_body_index.append([int(i) for i in tokens[1].split(",")])
        elif "ClickedBody" in tokens[0]:
            click_body_index.append([int(i) for i in tokens[1].split(",")])
        elif "CandidateVert" in tokens[0]:
            candidate_vert_index.append([int(tokens[1])])
        elif "ClickedVert" in tokens[0]:
            click_vert_index.append([int(tokens[1])])
        elif "CandidateSubvert" in tokens[0]:
            candidate_subvert_index.append([int(tokens[1])])
        elif "ClickedSubvert" in tokens[0]:
            click_subvert_index.append([int(tokens[1])])
        else:
            print(tokens[0])
            raise ValueError("data format is wrong")

    return [
        label,  #0
        imp_index, #1
        user_index, #2
        candidate_title_index, #3
        click_title_index, #4
        candidate_body_index, #5
        click_body_index, #6
        candidate_vert_index, #7
        click_vert_index, #8
        candidate_subvert_index, #9
        click_subvert_index #10
    ]

def load_all(test_data_file=False):
    max_user, max_item = 0, 0
    train_data = {}
    f = open(conf.train_data_path)
    for idx, line in enumerate(f):
        train_data[idx] = parser_one_line(line, conf.npratio)

    val_data = {}
    f = open(conf.val_data_path)
    for idx, line in enumerate(f):
        val_data[idx] = parser_one_line(line)
    
    if test_data_file:
        test_data = {}
        f = open(conf.test_data_path)
        for idx, line in enumerate(f):
            test_data[idx] = parser_one_line(line)
    
        return train_data, val_data, test_data
    else:
        return train_data, val_data
        
class TrainData():
    def __init__(self, train_data):
        self.train_data = train_data
        self.length = len(train_data.keys())

    def _get_batch(self, batch_idx_list):
        labels = []
        pred_input_title = []
        pred_input_body = []
        pred_input_vert = []
        pred_input_subvert = []
        his_input_title = []
        his_input_body = []
        his_input_vert = []
        his_input_subvert = []

        for batch_idx in batch_idx_list:
            labels.append(0)

            # get pred_input_title, pred_input_body, pred_input_vert, pred_input_subvert
            # [[b0_c0, b0_c1, ..., b0_cC], [b1_c0, b1_c1, ..., b1_cC], ...]
            for can_idx in range(conf.npratio+1):
                pred_input_title.append(self.train_data[batch_idx][3][can_idx])
                pred_input_body.append(self.train_data[batch_idx][5][can_idx])
                pred_input_vert.append(self.train_data[batch_idx][7][can_idx])
                pred_input_subvert.append(self.train_data[batch_idx][9][can_idx])

            # get his_input_title, his_input_body, his_input_vert, his_input_subvert
            # [[b0_h0, b0_h1, ..., b0_hN], [b1_h0, b1_h1, ..., b1_hN], ...]
            for his_idx in range(conf.his_size):
                his_input_title.append(self.train_data[batch_idx][4][his_idx])
                his_input_body.append(self.train_data[batch_idx][6][his_idx])
                his_input_vert.append(self.train_data[batch_idx][8][his_idx])
                his_input_subvert.append(self.train_data[batch_idx][10][his_idx])

        return [
            torch.LongTensor(pred_input_title).cuda(), 
            torch.LongTensor(pred_input_body).cuda(), 
            torch.LongTensor(pred_input_vert).cuda(), 
            torch.LongTensor(pred_input_subvert).cuda(),
        ], [
            torch.LongTensor(his_input_title).cuda(), 
            torch.LongTensor(his_input_body).cuda(), 
            torch.LongTensor(his_input_vert).cuda(),
            torch.LongTensor(his_input_subvert).cuda(),
        ], torch.LongTensor(labels).cuda()

class TestData():
    def __init__(self, test_data):
        self.test_data = test_data
        self.length = len(test_data.keys())

    def _get_batch(self, batch_idx_list):
        labels = []
        pred_input_title = []
        pred_input_body = []
        pred_input_vert = []
        pred_input_subvert = []
        his_input_title = []
        his_input_body = []
        his_input_vert = []
        his_input_subvert = []
        imp_indexes = []

        for batch_idx in batch_idx_list:
            pred_input_title.append(self.test_data[batch_idx][3][0])
            pred_input_body.append(self.test_data[batch_idx][5][0])
            pred_input_vert.append(self.test_data[batch_idx][7][0])
            pred_input_subvert.append(self.test_data[batch_idx][9][0])

            labels.append(self.test_data[batch_idx][0][0])
            imp_indexes.append(self.test_data[batch_idx][1][0])

            # get his_input_title, 
            # [[b0_h0, b0_h1, ..., b0_hN], [b1_h0, b1_h1, ..., b1_hN], ...]
            for his_idx in range(conf.his_size):
                his_input_title.append(self.test_data[batch_idx][4][his_idx])
                his_input_body.append(self.test_data[batch_idx][6][his_idx])
                his_input_vert.append(self.test_data[batch_idx][8][his_idx])
                his_input_subvert.append(self.test_data[batch_idx][10][his_idx])

        return [
            torch.LongTensor(pred_input_title).cuda(), 
            torch.LongTensor(pred_input_body).cuda(), 
            torch.LongTensor(pred_input_vert).cuda(), 
            torch.LongTensor(pred_input_subvert).cuda(),
        ], [
            torch.LongTensor(his_input_title).cuda(), 
            torch.LongTensor(his_input_body).cuda(), 
            torch.LongTensor(his_input_vert).cuda(),
            torch.LongTensor(his_input_subvert).cuda(),
        ], labels, imp_indexes