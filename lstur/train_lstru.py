import os, sys, shutil
import torch
import torch.nn as nn
import torch.utils.data as data
import numpy as np
from time import time

import DataModule_lstur as data_utils
import config_lstur as conf
from metrics import evaluate
from Logging import Logging, check_dir, tensorToScalar


if __name__ == '__main__':
    ############################## CREATE MODEL ##############################
    from lstur import lstur
    model = lstur()
    
    model.cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=conf.learning_rate)

    ############################## PREPARE DATASET ##############################
    print('System start to load data...')
    t0 = time()
    train_data, val_data = data_utils.load_all()
    t1 = time()
    print('Data has been loaded successfully, cost:%.4fs' % (t1 - t0))

    ########################### TRAINING STAGE ##################################
    check_dir('%s/train_log' % conf.out_path)
    log = Logging('%s/train_%s_lstur.log' % (conf.out_path, conf.data_name))
    train_model_path = '%s/train_%s_lstur.mod' % (conf.out_path, conf.data_name)

    # prepare data for the training stage
    train_dataset = data_utils.TrainData(train_data)
    val_dataset = data_utils.TestData(val_data)

    train_batch_sampler = data.BatchSampler(data.RandomSampler(
        range(train_dataset.length)), batch_size=conf.batch_size, drop_last=False)
    val_batch_sampler = data.BatchSampler(data.SequentialSampler(
        range(val_dataset.length)), batch_size=conf.batch_size, drop_last=True)

    # Start Training !!!
    max_auc = 0
    for epoch in range(1, conf.train_epochs+1):
        t0 = time()
        model.train()

        train_loss = []
        for batch_idx_list in train_batch_sampler:
            user_indexes, his_input_title, pred_input_title, labels = \
                train_dataset._get_batch(batch_idx_list)
            obj_loss = model(user_indexes, his_input_title, pred_input_title, labels)
            #import pdb; pdb.set_trace()
            train_loss.append(obj_loss.item())
            model.zero_grad(); obj_loss.backward(); optimizer.step()

        train_loss = np.mean(train_loss)

        t1 = time()

        # evaluate the performance of the model with following xxx 
        model.eval()

        val_preds, val_labels, val_keys = [], [], []
        for batch_idx_list in val_batch_sampler:
            user_indexes, his_input_title, pred_input_title, labels, keys = \
                val_dataset._get_batch(batch_idx_list)
            preds = model.predict(user_indexes, his_input_title, pred_input_title)    
            val_preds.extend(tensorToScalar(preds))
            val_labels.extend(labels)
            val_keys.extend(keys)

        #import pdb; pdb.set_trace()
        auc, mrr, ndcg_5, ndcg_10 = evaluate(val_labels, val_preds, val_keys)
        t2 = time()

        # save model when auc > max_auc
        if epoch == 1:
            max_auc = auc
        if auc > max_auc:
            torch.save(model.state_dict(), train_model_path)
        max_auc = max(auc, max_auc)

        log.record('Training Stage: Epoch:%d, compute loss cost:%.4fs, evaluation cost:%.4fs' % (epoch, (t1-t0), (t2-t1)))
        log.record('Train loss:{:.4f}'.format(train_loss))
        log.record('auc:%.4f, mrr:%.4f, ndcg@5:%.4f, ndcg@10:%.4f' % (auc, mrr, ndcg_5, ndcg_10))