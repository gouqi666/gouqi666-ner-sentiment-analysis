import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from transformers import  BertModel,BertTokenizer,AutoTokenizer, AutoModel
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
import torch
import os
from utils import load_vectors,ner_accuary
from preprocess import data_process,read_data_from_csv,bert_ner_dataset
from model import BERT_CRF
tokenizer = AutoTokenizer.from_pretrained("nghuyong/ernie-1.0")
def bert_collate_fn(batch,token_id=tokenizer.pad_token_id,token_type_id=tokenizer.pad_token_type_id):
    inputs_ids,token_type_ids,attention_mask,labels,text_lengths = zip(*batch)
    max_len = max([len(seq_a) for seq_a in inputs_ids]) #这里我使用的是一个batch中text_a或者是text_b的最大长度作为max_len,也可以自定义长度
    inputs_ids = [seq + [token_id]*(max_len-len(seq)) for seq in inputs_ids]
    token_type_ids = [seq + [token_type_id]*(max_len-len(seq)) for seq in token_type_ids]
    attention_mask = [seq + [0]*(max_len-len(seq)) for seq in attention_mask]
    labels =  [[-1] + seq + [-1]*(max_len-len(seq)-1) for seq in labels]  #cls 和sep
    return torch.tensor(inputs_ids,dtype = torch.long),torch.tensor(token_type_ids,dtype = torch.long),torch.tensor(attention_mask,dtype = torch.long),torch.tensor(labels,dtype=torch.long),torch.tensor(text_lengths,dtype=torch.long)

def seed_torch(seed=42):
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False

if __name__ == "__main__":
    seed_torch()
    train_data, test_data = read_data_from_csv()
    tokenizer = AutoTokenizer.from_pretrained("nghuyong/ernie-1.0")
    tag_to_ix = {'B-BANK': 0, 'I-BANK': 1, 'O': 2, 'B-COMMENTS_N': 3, 'I-COMMENTS_N': 4, 'B-COMMENTS_ADJ': 5,
                 'I-COMMENTS_ADJ': 6, 'B-PRODUCT': 7, 'I-PRODUCT': 8, '<START>': 9, '<STOP>': 10}
    ## datasets
    train_datasets = bert_ner_dataset(train_data['text'], train_data['BIO_anno'], tag_to_ix)
    # train_datasets, valid_datasets = torch.utils.data.random_split(train_datasets, [5522, 500],
    #                                                                generator=torch.Generator().manual_seed(42))
    train_data_loader = DataLoader(
        train_datasets,
        batch_size=64,
        collate_fn=bert_collate_fn,
        shuffle=True
    )
    # valid_data_loader = DataLoader(
    #     valid_datasets,
    #     batch_size=32,
    #     collate_fn=bert_collate_fn,
    #     shuffle=True
    # )
    ## train parameters
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)

    model = BERT_CRF(tag_to_ix)
    criterion = nn.CrossEntropyLoss(ignore_index=-1)
    criterion = criterion.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-5)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[20, 80], gamma=0.95)
    model = model.to(device)
    ckpt_dir = './'

    # train
    epoches = 30
    global_step = 0
    valid_F1 = float('inf')
    min_loss = float('inf')
    for epoch in range(1, epoches + 1):
        model.train()
        train_loss = []
        for step, batch in enumerate(train_data_loader, start=1):
            optimizer.zero_grad()
            input_ids, token_type_ids, attention_mask, labels,text_lengths = batch
            input_ids = input_ids.to(device, dtype=torch.long)
            token_type_ids = token_type_ids.to(device, dtype=torch.long)
            attention_mask = attention_mask.to(device, dtype=torch.long)
            labels = labels.to(device, dtype=torch.long)
            text_lengths = text_lengths.to(device,dtype=torch.long)
            # 喂数据给model
            # crf
            loss = model.forward(input_ids,token_type_ids,attention_mask,labels,text_lengths)
            # 计算损失函数值
            train_loss.append(loss)
            loss.backward()
            print('global step: %d,loss:%.5f:' %(global_step, loss))
            optimizer.step()
            scheduler.step()
            global_step += 1
        mean_loss = sum(train_loss) / len(train_loss)
        print('epoch:%d,mean_loss:%.5f' % (
        epoch, mean_loss))
        # 评估当前训练的模型
        model.eval()
        save_dir = os.path.join(ckpt_dir, "model")
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        if mean_loss < min_loss:
            min_loss = mean_loss
            torch.save(model.state_dict(), os.path.join(save_dir, 'ner_bert_crf_best_model.pt'))
        # with torch.no_grad():
        #     torch.cuda.empty_cache()
        #     total_F1 = []
        #     for i, batch in enumerate(valid_data_loader):
        #         input_ids, token_type_ids, attention_mask, labels, text_lengths = batch
        #         input_ids = input_ids.to(device, dtype=torch.long)
        #         token_type_ids = token_type_ids.to(device, dtype=torch.long)
        #         attention_mask = attention_mask.to(device, dtype=torch.long)
        #         labels = labels.to(device, dtype=torch.long)
        #         text_lengths = text_lengths.to(device, dtype=torch.long)
        #         # 喂数据给model
        #         pred = model.decode(input_ids, token_type_ids, attention_mask, text_lengths)
        #         p, r, F1 = ner_accuary(pred, labels[:, 1:], tag_to_ix, text_lengths)
        #         total_F1.append(F1)
        #         tags = []
        #         ret = []
        #         id_to_tag = list(tag_to_ix.keys())
        #         # 展示每个batch的第一个数据结果
        #         for id in labels[0][1:text_lengths[0]+1]:
        #             tags.append(id_to_tag[id])
        #         for id in pred[0][:text_lengths[0]]:
        #             ret.append(id_to_tag[id])
        #         print('orgin-tag:', tags)
        #         print('predict-tag:', ret)
        #         print("F1 score:", F1)
        #     cur_F1 = sum(total_F1) / len(total_F1)
        #     print("valid F1 score:",cur_F1)
        #     if cur_F1 > valid_F1:
        #         valid_F1 = cur_F1
        #         torch.save(model.state_dict(), os.path.join(save_dir, 'ner_bert_crf2_best_model.pt'))