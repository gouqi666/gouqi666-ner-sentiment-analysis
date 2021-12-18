import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from transformers import  BertModel,BertTokenizer
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
import torch
import os
from utils import load_vectors
from preprocess import data_process,read_data_from_csv,ner_dataset,bert_ner_dataset
from model import BERT_CRF
tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
def bert_collate_fn(batch,token_id=tokenizer.pad_token_id,token_type_id=tokenizer.pad_token_type_id):
    inputs_ids,token_type_ids,attention_mask,labels,text_lengths = zip(*batch)
    max_len = max([len(seq_a) for seq_a in inputs_ids]) #这里我使用的是一个batch中text_a或者是text_b的最大长度作为max_len,也可以自定义长度
    inputs_ids = [seq + [token_id]*(max_len-len(seq)) for seq in inputs_ids]
    token_type_ids = [seq + [token_type_id]*(max_len-len(seq)) for seq in token_type_ids]
    attention_mask = [seq + [0]*(max_len-len(seq)) for seq in attention_mask]
    labels =  [[-1] + seq + [-1]*(max_len-len(seq)-1) for seq in labels]  #cls 和sep
    return torch.tensor(inputs_ids,dtype = torch.long),torch.tensor(token_type_ids,dtype = torch.long),torch.tensor(attention_mask,dtype = torch.long),torch.tensor(labels,dtype=torch.long),torch.tensor(text_lengths,dtype=torch.long)




if __name__ == "__main__":

    train_data, test_data = read_data_from_csv()
    tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
    tag_to_ix = {'B-BANK': 0, 'I-BANK': 1, 'O': 2, 'B-COMMENTS_N': 3, 'I-COMMENTS_N': 4, 'B-COMMENTS_ADJ': 5,
                 'I-COMMENTS_ADJ': 6, 'B-PRODUCT': 7, 'I-PRODUCT': 8, '<START>': 9, '<STOP>': 10}
    ## datasets
    train_datasets = bert_ner_dataset(train_data['text'], train_data['BIO_anno'], tag_to_ix)
    train_datasets, valid_datasets = torch.utils.data.random_split(train_datasets, [7000, 528],
                                                                   generator=torch.Generator().manual_seed(42))
    train_data_loader = DataLoader(
        train_datasets,
        batch_size=32,
        collate_fn=bert_collate_fn,
        shuffle=True
    )
    valid_data_loader = DataLoader(
        valid_datasets,
        batch_size=32,
        collate_fn=bert_collate_fn,
        shuffle=True
    )
    ## train parameters
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)

    model = BERT_CRF(tag_to_ix)
    criterion = nn.CrossEntropyLoss(ignore_index=-1)
    criterion = criterion.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-5)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[20, 80], gamma=0.4)
    model = model.to(device)
    ckpt_dir = './'

    # train
    epoches = 10
    global_step = 0
    for epoch in range(1, epoches + 1):
        model.train()
        train_loss = 0
        for step, batch in enumerate(train_data_loader, start=1):
            optimizer.zero_grad()
            input_ids, token_type_ids, attention_mask, labels,text_lengths = batch
            input_ids = input_ids.to(device, dtype=torch.long)
            token_type_ids = token_type_ids.to(device, dtype=torch.long)
            attention_mask = attention_mask.to(device, dtype=torch.long)
            labels = labels.to(device, dtype=torch.long)
            text_lengths = text_lengths.to(device,dtype=torch.long)
            # 喂数据给model
            # pred = model.forward(input_ids,token_type_ids,attention_mask,labels,text_lengths)
            # loss = 0
            # for i in range(pred.size(0)):
            #     loss += criterion(pred[i],labels[i])

            # crf
            loss = model.forward(input_ids,token_type_ids,attention_mask,labels,text_lengths)

            # 计算损失函数值
            train_loss += loss
            loss.backward()
            print('loss:',loss)
            optimizer.step()
            scheduler.step()
            global_step += 1
        print('epoch:%d,mean_loss:%.5f' % (
        epoch, train_loss / len(train_data_loader)))
        # 评估当前训练的模型
        model.eval()
        save_dir = os.path.join(ckpt_dir, "model")
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        with torch.no_grad():
            torch.cuda.empty_cache()
            total_acc = []
            for i, batch in enumerate(valid_data_loader):
                input_ids, token_type_ids, attention_mask, labels, text_lengths = batch
                input_ids = input_ids.to(device, dtype=torch.long)
                token_type_ids = token_type_ids.to(device, dtype=torch.long)
                attention_mask = attention_mask.to(device, dtype=torch.long)
                labels = labels.to(device, dtype=torch.long)
                text_lengths = text_lengths.to(device, dtype=torch.long)
                # 喂数据给model
                # pred = model.forward(input_ids, token_type_ids, attention_mask, labels, text_lengths)
                # pred = torch.argmax(pred,-1)
                pred = model.decode(input_ids, token_type_ids, attention_mask, text_lengths)
                tags = []
                ret = []
                id_to_tag = list(tag_to_ix.keys())
                for id in labels[0][1:text_lengths[0]]:
                    tags.append(id_to_tag[id])
                for id in pred[0][1:text_lengths[0]]:
                    ret.append(id_to_tag[id])
                print('orgin-tag:', tags)
                print('predict-tag:', ret)
            torch.save(model.state_dict(), os.path.join(save_dir, 'ner_bert_best_model.pt'))




