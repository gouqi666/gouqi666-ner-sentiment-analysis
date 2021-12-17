import torch
from torch.utils.data import Dataset
from torch.utils.data import random_split
import time
import torch.nn as nn
from torch import optim
from torch.optim import lr_scheduler
from model import sentiment_model
from transformers import  BertModel,BertTokenizer,AutoTokenizer, AutoModel
from preprocess import sentiment_dataset,read_data_from_csv
import os
from torch.utils.data import DataLoader
import numpy as np

tokenizer = AutoTokenizer.from_pretrained("nghuyong/ernie-1.0")
def sentiment_collate_fn(batch_data,token_id=tokenizer.pad_token_id,token_type_id=tokenizer.pad_token_type_id):
    inputs_ids,token_type_ids,attention_mask,label = list(zip(*(batch_data)))#batch_data的结构是[([texta_1],[textb_1],[label_1]),([texta_2],[textb_2],[label_2])，...]，所以需要使用zip函数对它解压
    max_len = max([len(seq_a) for seq_a in inputs_ids]) #这里我使用的是一个batch中text_a或者是text_b的最大长度作为max_len,也可以自定义长度
    inputs_ids = [seq + [token_id]*(max_len-len(seq)) for seq in inputs_ids]
    token_type_ids = [seq + [token_type_id]*(max_len-len(seq)) for seq in token_type_ids]
    attention_mask = [seq + [0]*(max_len-len(seq)) for seq in attention_mask]
    return torch.tensor(inputs_ids,dtype = torch.long),torch.tensor(token_type_ids,dtype = torch.long),torch.tensor(attention_mask,dtype = torch.long),torch.tensor(label,dtype=torch.long)

def seed_torch(seed=42):
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False



if __name__ == '__main__':
    seed_torch()
    ##########    dataset
    # print(len(train_data)) # train_data 总共7528条数据
    train_data,test_data = read_data_from_csv()
    train_dataset = sentiment_dataset(train_data['text'],train_data['class'])
    # train_dataset,valid_dataset = random_split(train_dataset,[7000,528],generator=torch.Generator().manual_seed(42))
    train_data_loader = DataLoader(
                    train_dataset,
                    batch_size = 32,
                    collate_fn = sentiment_collate_fn,
                    shuffle=True
    )
    # valid_data_loader = DataLoader(
    #                 valid_dataset,
    #                 batch_size = 32,
    #                 collate_fn = sentiment_collate_fn,
    #                 shuffle=True
    # )


    ## train parameters
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)

    model = sentiment_model()
    criterion = nn.CrossEntropyLoss()
    criterion = criterion.to(device)
    optimizer = optim.Adam(model.parameters(), lr=2e-5)
    scheduler = lr_scheduler.MultiStepLR(optimizer,milestones=[20,80],gamma = 0.95)
    model = model.to(device)
    ckpt_dir = './'

    ######### train
    # train
    epoches = 20
    global_step = 0
    tic_train = time.time()
    min_loss = float('inf')
    for epoch in range(1, epoches + 1):
        model.train()
        train_loss = 0
        total_acc = []
        for step, batch in enumerate(train_data_loader, start=1):
            optimizer.zero_grad()
            input_ids, token_type_ids, attention_mask, labels = batch
            input_ids = input_ids.to(device, dtype=torch.long)
            token_type_ids = token_type_ids.to(device, dtype=torch.long)
            attention_mask = attention_mask.to(device, dtype=torch.long)
            labels = labels.to(device, dtype=torch.long)
            # 喂数据给model
            probs = model(input_ids=input_ids,token_type_ids = token_type_ids, attention_mask = attention_mask)
            # 计算损失函数值
            loss = criterion(probs, labels)
            train_loss += loss
            loss.backward()
            optimizer.step()
            scheduler.step()
            # 预测分类概率值
            # 计算acc
            batch_acc = sum(torch.argmax(probs,-1).squeeze() == labels) / len(labels)
            total_acc.append(batch_acc.item())
            global_step += 1
            if global_step % 30 == 0:
                #             acc = train_accuracy.compute()
                print(
                    "global step %d, epoch: %d, batch: %d, loss: %.5f, accu: %.5f, speed: %.2f step/s"
                    % (global_step, epoch, step, loss, batch_acc,
                       10 / (time.time() - tic_train)))
                tic_train = time.time()
        mean_loss = sum(total_acc) / len(total_acc)

        print('epoch:%d,total_loss:%.5f,accu:%.5f' % (epoch, train_loss * epoch / global_step,mean_loss) )
        save_dir = os.path.join(ckpt_dir, "model")
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        if mean_loss < min_loss:
            torch.save(model.state_dict(), os.path.join(save_dir, 'sentiment_best_model.pt'))
        # 评估当前训练的模型
        # model.eval()
        # save_dir = os.path.join(ckpt_dir, "model")
        # if not os.path.exists(save_dir):
        #     os.makedirs(save_dir)
        # with torch.no_grad():
        #     torch.cuda.empty_cache()
        #     total_acc = []
        #     for i, batch in enumerate(valid_data_loader):
        #         input_ids, token_type_ids, attention_mask, labels = batch
        #         input_ids = input_ids.to(device, dtype=torch.long)
        #         token_type_ids = token_type_ids.to(device, dtype=torch.long)
        #         attention_mask = attention_mask.to(device, dtype=torch.long)
        #         labels = labels.to(device, dtype=torch.long)
        #         probs = model(input_ids=input_ids,token_type_ids = token_type_ids, attention_mask = attention_mask)
        #         batch_acc = sum(torch.argmax(probs,-1).squeeze() == labels) / len(labels)
        #         total_acc.append(batch_acc)
        #     acc = sum(total_acc) / len(total_acc)
        #     print('eval_acc: %.5f ' % acc)
        #     torch.save(model.state_dict(), os.path.join(save_dir, 'sentiment_best_model.pt'))