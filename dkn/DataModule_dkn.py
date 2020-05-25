import torch
import config_naml as conf

# refer to https://github.com/microsoft/recommenders/blob/master/reco_utils/recommender/newsrec/io/naml_iterator.py
def parser_one_line(line, ID_spliter='%', col_spliter=' ', ):
    impression_id = None
    words = line.strip().split(ID_spliter)
    if len(words) == 2:
        impression_id = words[1].strip()

    cols = words[0].strip().split(col_spliter)
    label = float(cols[0])
    candidate_news_index = []
    #candidate_news_val = []
    click_news_index = []
    #click_news_val = []
    candidate_news_entity_index = []
    click_news_entity_index = []

    for news in cols[1:]:
        tokens = news.split(":")
        if tokens[0] == "CandidateNews":
            # word index start by 0
            for item in tokens[1].split(","):
                candidate_news_index.append(int(item))
                #candidate_news_val.append(float(1))
        elif "clickedNews" in tokens[0]:
            tmp_click_news_index = []
            for item in tokens[1].split(","):
                click_news_index.append(int(item))
                #click_news_val.append(float(1))

        elif tokens[0] == "entity":
            for item in tokens[1].split(","):
                candidate_news_entity_index.append(int(item))
        elif "entity" in tokens[0]:
            for item in tokens[1].split(","):
                click_news_entity_index.append(int(item))

        else:
            raise ValueError("data format is wrong")

    return [
        label,
        candidate_news_index,
        #candidate_news_val,
        click_news_index,
        #click_news_val,
        candidate_news_entity_index,
        click_news_entity_index,
        impression_id,
    ]

def load_all(test_data_file=False):
    max_user, max_item = 0, 0
    train_data = {}
    f = open(conf.train_data_path)
    for idx, line in enumerate(f):
        train_data[idx] = parser_one_line(line)

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
        pred_input_entity = []
        his_input_title = []
        his_input_entity = []

        for batch_idx in batch_idx_list:
            labels.append(0)

            # get pred_input_title, pred_input_body, pred_input_vert, pred_input_subvert
            # [[b0_c0, b0_c1, ..., b0_cC], [b1_c0, b1_c1, ..., b1_cC], ...]
            pred_input_title.append(self.train_data[batch_idx][3][can_idx])
            pred_input_body.append(self.train_data[batch_idx][5][can_idx])

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