import torch
import torch.nn as nn
import os
from utils import load_vectors
from preprocess import read_data_from_csv,data_process
from torch.utils.data import DataLoader
from model import lstm_ner_dataset,bert_ner_dataset,BERT_CRF, BiLSTM_CRF,BERT,BERT_CRF2
from utils import ner_accuary
from transformers import  BertModel,BertTokenizer,AutoTokenizer, AutoModel
tokenizer = AutoTokenizer.from_pretrained("nghuyong/ernie-1.0")

def lstm_ner_collate_fn(batch):
    inputs_ids,label_ids,text_lengths = zip(*batch)
    max_len = max([length for length in text_lengths])
    inputs_ids = [seq + [0]*(max_len-len(seq)) for seq in inputs_ids]
    label_ids = [seq + [0]*(max_len-len(seq)) for seq in label_ids]
    return torch.tensor(inputs_ids,dtype = torch.long),torch.tensor(label_ids,dtype=torch.long),torch.tensor(text_lengths,dtype=torch.long)

def lstm_ner_test_collate_fn(batch):
    inputs_ids,text_lengths = zip(*batch)
    max_len = max([length for length in text_lengths])
    inputs_ids = [seq + [0]*(max_len-len(seq)) for seq in inputs_ids]
    return torch.tensor(inputs_ids,dtype = torch.long),torch.tensor(text_lengths,dtype=torch.long)


def bert_collate_fn(batch,token_id=tokenizer.pad_token_id,token_type_id=tokenizer.pad_token_type_id):
    inputs_ids,token_type_ids,attention_mask,labels,text_lengths = zip(*batch)
    max_len = max([len(seq_a) for seq_a in inputs_ids]) #这里我使用的是一个batch中text_a或者是text_b的最大长度作为max_len,也可以自定义长度
    inputs_ids = [seq + [token_id]*(max_len-len(seq)) for seq in inputs_ids]
    token_type_ids = [seq + [token_type_id]*(max_len-len(seq)) for seq in token_type_ids]
    attention_mask = [seq + [0]*(max_len-len(seq)) for seq in attention_mask]
    labels =  [[-1] + seq + [-1]*(max_len-len(seq)-1) for seq in labels]  #cls 和sep
    return torch.tensor(inputs_ids,dtype = torch.long),torch.tensor(token_type_ids,dtype = torch.long),torch.tensor(attention_mask,dtype = torch.long),torch.tensor(labels,dtype=torch.long),torch.tensor(text_lengths,dtype=torch.long)


def bert_test_collate_fn(batch,token_id=tokenizer.pad_token_id,token_type_id=tokenizer.pad_token_type_id):
    inputs_ids,token_type_ids,attention_mask,text_lengths = zip(*batch)
    max_len = max([len(seq_a) for seq_a in inputs_ids]) #这里我使用的是一个batch中text_a或者是text_b的最大长度作为max_len,也可以自定义长度
    inputs_ids = [seq + [token_id]*(max_len-len(seq)) for seq in inputs_ids]
    token_type_ids = [seq + [token_type_id]*(max_len-len(seq)) for seq in token_type_ids]
    attention_mask = [seq + [0]*(max_len-len(seq)) for seq in attention_mask]
    return torch.tensor(inputs_ids,dtype = torch.long),torch.tensor(token_type_ids,dtype = torch.long),torch.tensor(attention_mask,dtype = torch.long),torch.tensor(text_lengths,dtype=torch.long)

class BiLSTM_CRF_TRAINER:
    def __init__(self):
        fast_text, word2idx = load_vectors("../../ff/cc.zh.300.vec")  # fast_text词向量位置
        train_data, test_data = read_data_from_csv()
        self.word2idx = word2idx
        ## datasets
        input_ids, label_ids, tag_to_ix, text_lengths = data_process(train_data['text'], train_data['BIO_anno'],
                                                                     word2idx=word2idx)
        vocab_size = len(word2idx)
        self.tag_to_ix = {'B-BANK': 0, 'I-BANK': 1, 'O': 2, 'B-COMMENTS_N': 3, 'I-COMMENTS_N': 4, 'B-COMMENTS_ADJ': 5,
                     'I-COMMENTS_ADJ': 6, 'B-PRODUCT': 7, 'I-PRODUCT': 8, '<START>': 9, '<STOP>': 10}
        embedding = torch.tensor(fast_text, dtype=torch.float)
        hidden_dim = 100
        self.model = BiLSTM_CRF(vocab_size, tag_to_ix, embedding, hidden_dim)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=3e-3)
        self.train_datasets = lstm_ner_dataset(input_ids, label_ids, text_lengths)
        # train_datasets, valid_datasets = torch.utils.data.random_split(train_datasets, [5522, 500],
        #                                                                generator=torch.Generator().manual_seed(42))
        self.train_data_loader = DataLoader(self.train_datasets, batch_size=64, num_workers=4, collate_fn=lstm_ner_collate_fn,
                                       shuffle=True)
        # self.valid_data_loader = DataLoader(valid_datasets, batch_size=32, num_workers=4, collate_fn=lstm_ner_collate_fn,
        #                                shuffle=True)


    def train(self):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        epochs = 10
        step = 0
        model = self.model
        train_data_loader = self.train_data_loader
        optimizer = self.optimizer
        tag_to_ix = self.tag_to_ix
        # valid_data_loader = self.valid_data_loader
        model.to(device)
        min_loss = float('inf')
        for epoch in range(epochs):
            model.train()
            train_loss = []
            for batch in train_data_loader:
                optimizer.zero_grad()
                X, label, text_lengths = batch
                X = X.to(device)
                label = label.to(device)
                text_lengths = text_lengths.to(device)
                loss = model.forward(X, label, text_lengths)
                train_loss.append(loss)
                ret = model.decode(X, text_lengths, tag_to_ix)
                print('epoch:%d,step:%d,loss:%.5f' % (epoch, step, loss))
                step += 1
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=20, norm_type=2)
                optimizer.step()
            mean_loss = sum(train_loss) / len(train_loss)
            print('epoch:%d,mean_loss:%.5f' % (
                epoch, mean_loss))
            # 评估当前训练的模型
            model.eval()
            if mean_loss < min_loss:
                min_loss = mean_loss
                torch.save(model.state_dict(),'./model/ner_lstm_best_model.pt')

    def predict(self):
        _, test_data = read_data_from_csv()
        tag_to_ix = self.tag_to_ix
        model = self.model
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        input_ids,text_lengths = data_process(test_data['text'],word2idx=self.word2idx,is_train=False)
        test_datasets = lstm_ner_dataset(input_ids, text_lengths = text_lengths, is_train=False)
        test_data_loader = DataLoader(
            test_datasets,
            batch_size=32,
            collate_fn=lstm_ner_test_collate_fn,
            shuffle=False
        )
        model.load_state_dict(torch.load('./model/ner_lstm_best_model.pt'))
        model = model.to(device)
        BIO = []
        model.eval()
        torch.cuda.empty_cache()
        model.eval()
        id_to_tag = list(tag_to_ix.keys())
        for batch in test_data_loader:
            X,text_lengths = batch
            X = X.to(device)
            text_lengths = text_lengths.to(device)
            pred = model.decode(X, text_lengths, tag_to_ix)
            for pre, length in zip(pred, text_lengths):
                pre = pre[:length]
                ret = []
                for id in pre:
                    ret.append(id_to_tag[id])
                BIO.append(' '.join(ret))
        return BIO





class BERT_CRF_TRAINER:
    def __init__(self):
        train_data, test_data = read_data_from_csv()
        tokenizer = AutoTokenizer.from_pretrained("nghuyong/ernie-1.0")
        self.tag_to_ix = {'B-BANK': 0, 'I-BANK': 1, 'O': 2, 'B-COMMENTS_N': 3, 'I-COMMENTS_N': 4, 'B-COMMENTS_ADJ': 5,
                     'I-COMMENTS_ADJ': 6, 'B-PRODUCT': 7, 'I-PRODUCT': 8, '<START>': 9, '<STOP>': 10}
        ## datasets
        train_datasets = bert_ner_dataset(train_data['text'], train_data['BIO_anno'], self.tag_to_ix)
        # train_datasets, valid_datasets = torch.utils.data.random_split(train_datasets, [5522, 500],
        #                                                                generator=torch.Generator().manual_seed(42))
        self.train_data_loader = DataLoader(
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

        self.model = BERT_CRF(self.tag_to_ix)
        self.criterion = nn.CrossEntropyLoss(ignore_index=-1)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=2e-5)
        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=[20, 80], gamma=0.95)
    def train(self):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(device)

        model = self.model
        criterion = self.criterion
        optimizer = self.optimizer
        scheduler = self.scheduler
        train_data_loader = self.train_data_loader
        model = model.to(device)
        criterion = criterion.to(device)

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
                input_ids, token_type_ids, attention_mask, labels, text_lengths = batch
                input_ids = input_ids.to(device, dtype=torch.long)
                token_type_ids = token_type_ids.to(device, dtype=torch.long)
                attention_mask = attention_mask.to(device, dtype=torch.long)
                labels = labels.to(device, dtype=torch.long)
                text_lengths = text_lengths.to(device, dtype=torch.long)
                # 喂数据给model
                # crf
                loss = model.forward(input_ids, token_type_ids, attention_mask, labels, text_lengths)
                # 计算损失函数值
                train_loss.append(loss)
                loss.backward()
                print('global step: %d,loss:%.5f:' % (global_step, loss))
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

    def predict(self):
        tag_to_ix = self.tag_to_ix
        _, test_data = read_data_from_csv()
        test_datasets = bert_ner_dataset(test_data['text'], tag_to_ix, is_train=False)
        test_data_loader = DataLoader(
            test_datasets,
            batch_size=32,
            collate_fn=bert_test_collate_fn,
            shuffle=False
        )
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        model = self.model
        model = model.to(device)
        model.load_state_dict(torch.load('./model/ner_bert_crf_best_model.pt'))
        BIO = []
        model.eval()
        torch.cuda.empty_cache()
        for i, batch in enumerate(test_data_loader):
            input_ids, token_type_ids, attention_mask, text_lengths = batch
            input_ids = input_ids.to(device, dtype=torch.long)
            token_type_ids = token_type_ids.to(device, dtype=torch.long)
            attention_mask = attention_mask.to(device, dtype=torch.long)
            text_lengths = text_lengths.to(device, dtype=torch.long)
            # 喂数据给model
            pred = model.decode(input_ids, token_type_ids, attention_mask, text_lengths)
            id_to_tag = list(tag_to_ix.keys())
            for pre, length in zip(pred, text_lengths):
                pre = pre[:length]
                ret = []
                for id in pre:
                    ret.append(id_to_tag[id])
                BIO.append(' '.join(ret))
        return BIO

class BERT_CRF2_TRAINER:
    def __init__(self):
        train_data, test_data = read_data_from_csv()
        tokenizer = AutoTokenizer.from_pretrained("nghuyong/ernie-1.0")
        self.tag_to_ix = {'B-BANK': 0, 'I-BANK': 1, 'O': 2, 'B-COMMENTS_N': 3, 'I-COMMENTS_N': 4, 'B-COMMENTS_ADJ': 5,
                     'I-COMMENTS_ADJ': 6, 'B-PRODUCT': 7, 'I-PRODUCT': 8, '<START>': 9, '<STOP>': 10}
        ## datasets
        train_datasets = bert_ner_dataset(train_data['text'], train_data['BIO_anno'], self.tag_to_ix)
        # train_datasets, valid_datasets = torch.utils.data.random_split(train_datasets, [5522, 500],
        #                                                                generator=torch.Generator().manual_seed(42))
        self.train_data_loader = DataLoader(
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

        self.model = BERT_CRF2(self.tag_to_ix)
        self.criterion = nn.CrossEntropyLoss(ignore_index=-1)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=2e-5)
        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=[20, 80], gamma=0.95)
    def train(self):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(device)

        model = self.model
        criterion = self.criterion
        optimizer = self.optimizer
        scheduler = self.scheduler
        train_data_loader = self.train_data_loader
        model = model.to(device)
        criterion = criterion.to(device)

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
                input_ids, token_type_ids, attention_mask, labels, text_lengths = batch
                input_ids = input_ids.to(device, dtype=torch.long)
                token_type_ids = token_type_ids.to(device, dtype=torch.long)
                attention_mask = attention_mask.to(device, dtype=torch.long)
                labels = labels.to(device, dtype=torch.long)
                text_lengths = text_lengths.to(device, dtype=torch.long)
                # 喂数据给model
                # crf
                loss = model.forward(input_ids, token_type_ids, attention_mask, labels, text_lengths)
                # 计算损失函数值
                train_loss.append(loss)
                loss.backward()
                print('global step: %d,loss:%.5f:' % (global_step, loss))
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
                torch.save(model.state_dict(), os.path.join(save_dir, 'ner_bert_crf2_best_model.pt'))
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

    def predict(self):
        tag_to_ix = self.tag_to_ix
        _, test_data = read_data_from_csv()
        test_datasets = bert_ner_dataset(test_data['text'], tag_to_ix, is_train=False)
        test_data_loader = DataLoader(
            test_datasets,
            batch_size=32,
            collate_fn=bert_test_collate_fn,
            shuffle=False
        )
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        model = self.model
        model = model.to(device)
        model.load_state_dict(torch.load('./model/ner_bert_crf2_best_model.pt'))
        BIO = []
        model.eval()
        torch.cuda.empty_cache()
        for i, batch in enumerate(test_data_loader):
            input_ids, token_type_ids, attention_mask, text_lengths = batch
            input_ids = input_ids.to(device, dtype=torch.long)
            token_type_ids = token_type_ids.to(device, dtype=torch.long)
            attention_mask = attention_mask.to(device, dtype=torch.long)
            text_lengths = text_lengths.to(device, dtype=torch.long)
            # 喂数据给model
            pred = model.decode(input_ids, token_type_ids, attention_mask, text_lengths)
            id_to_tag = list(tag_to_ix.keys())
            for pre, length in zip(pred, text_lengths):
                pre = pre[:length]
                ret = []
                for id in pre:
                    ret.append(id_to_tag[id])
                BIO.append(' '.join(ret))
        return BIO

class BERT_TRAINER:
    def __init__(self):
        train_data, test_data = read_data_from_csv()
        tokenizer = AutoTokenizer.from_pretrained("nghuyong/ernie-1.0")
        self.tag_to_ix = {'B-BANK': 0, 'I-BANK': 1, 'O': 2, 'B-COMMENTS_N': 3, 'I-COMMENTS_N': 4, 'B-COMMENTS_ADJ': 5,
                     'I-COMMENTS_ADJ': 6, 'B-PRODUCT': 7, 'I-PRODUCT': 8, '<START>': 9, '<STOP>': 10}
        ## datasets
        train_datasets = bert_ner_dataset(train_data['text'], train_data['BIO_anno'], self.tag_to_ix)
        # train_datasets, valid_datasets = torch.utils.data.random_split(train_datasets, [5522, 500],
        #                                                                generator=torch.Generator().manual_seed(42))
        self.train_data_loader = DataLoader(
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

        self.model = BERT(self.tag_to_ix)
        self.criterion = nn.CrossEntropyLoss(ignore_index=-1)   # 忽略cls和后面的padding部分
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=2e-5)
        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=[20, 80], gamma=0.95)

    def train(self):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(device)

        model = self.model
        criterion = self.criterion
        optimizer = self.optimizer
        scheduler = self.scheduler
        train_data_loader = self.train_data_loader
        model = model.to(device)
        criterion = criterion.to(device)

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
                input_ids, token_type_ids, attention_mask, labels, text_lengths = batch
                input_ids = input_ids.to(device, dtype=torch.long)
                token_type_ids = token_type_ids.to(device, dtype=torch.long)
                attention_mask = attention_mask.to(device, dtype=torch.long)
                labels = labels.to(device, dtype=torch.long)
                text_lengths = text_lengths.to(device, dtype=torch.long)
                # 喂数据给model
                logits = model.forward(input_ids, token_type_ids, attention_mask)
                loss = criterion(torch.transpose(logits,1,2),labels) # N * C(多少个类) * T（序列长度）
                # 计算损失函数值
                train_loss.append(loss)
                loss.backward()
                print('global step: %d,loss:%.5f:' % (global_step, loss))
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
                torch.save(model.state_dict(), os.path.join(save_dir, 'ner_bert_best_model.pt'))

    def predict(self):
        tag_to_ix = self.tag_to_ix
        _, test_data = read_data_from_csv()
        test_datasets = bert_ner_dataset(test_data['text'], tag_to_ix, is_train=False)
        test_data_loader = DataLoader(
            test_datasets,
            batch_size=32,
            collate_fn=bert_test_collate_fn,
            shuffle=False
        )
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        model = self.model
        model = model.to(device)
        model.load_state_dict(torch.load('./model/ner_bert_best_model.pt'))
        BIO = []
        model.eval()
        torch.cuda.empty_cache()
        for i, batch in enumerate(test_data_loader):
            input_ids, token_type_ids, attention_mask, text_lengths = batch
            input_ids = input_ids.to(device, dtype=torch.long)
            token_type_ids = token_type_ids.to(device, dtype=torch.long)
            attention_mask = attention_mask.to(device, dtype=torch.long)
            text_lengths = text_lengths.to(device, dtype=torch.long)
            # 喂数据给model
            logits = model(input_ids, token_type_ids, attention_mask)
            pred = torch.argmax(logits,-1)
            id_to_tag = list(tag_to_ix.keys())
            for pre, length in zip(pred, text_lengths):
                pre = pre[1:length+1]
                ret = []
                for id in pre:
                    ret.append(id_to_tag[id])
                BIO.append(' '.join(ret))
        return BIO