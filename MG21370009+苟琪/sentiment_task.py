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


class FocalLoss_Ori(nn.Module):
    """
    This is a implementation of Focal Loss with smooth label cross entropy supported which is proposed in
    'Focal Loss for Dense Object Detection. (https://arxiv.org/abs/1708.02002)'
    Focal_Loss= -1*alpha*((1-pt)**gamma)*log(pt)
    Args:
        num_class: number of classes
        alpha: class balance factor
        gamma:
        ignore_index:
        reduction:
    """

    def __init__(self, num_class, alpha=None, gamma=2, ignore_index=None, reduction='mean'):
        super(FocalLoss_Ori, self).__init__()
        self.num_class = num_class
        self.gamma = gamma
        self.reduction = reduction
        self.smooth = 1e-4
        self.ignore_index = ignore_index
        self.alpha = alpha
        if alpha is None:
            self.alpha = torch.ones(num_class, )
        elif isinstance(alpha, (int, float)):
            self.alpha = torch.as_tensor([alpha] * num_class)
        elif isinstance(alpha, (list, np.ndarray)):
            self.alpha = torch.as_tensor(alpha)
        if self.alpha.shape[0] != num_class:
            raise RuntimeError('the length not equal to number of class')

        # if isinstance(self.alpha, (list, tuple, np.ndarray)):
        #     assert len(self.alpha) == self.num_class
        #     self.alpha = torch.Tensor(list(self.alpha))
        # elif isinstance(self.alpha, (float, int)):
        #     assert 0 < self.alpha < 1.0, 'alpha should be in `(0,1)`)'
        #     assert balance_index > -1
        #     alpha = torch.ones((self.num_class))
        #     alpha *= 1 - self.alpha
        #     alpha[balance_index] = self.alpha
        #     self.alpha = alpha
        # elif isinstance(self.alpha, torch.Tensor):
        #     self.alpha = self.alpha
        # else:
        #     raise TypeError('Not support alpha type, expect `int|float|list|tuple|torch.Tensor`')

    def forward(self, logit, target):
        # assert isinstance(self.alpha,torch.Tensor)\
        N, C = logit.shape[:2]
        alpha = self.alpha.to(logit.device)
        prob = F.softmax(logit, dim=1)
        if prob.dim() > 2:
            # N,C,d1,d2 -> N,C,m (m=d1*d2*...)
            prob = prob.view(N, C, -1)
            prob = prob.transpose(1, 2).contiguous()  # [N,C,d1*d2..] -> [N,d1*d2..,C]
            prob = prob.view(-1, prob.size(-1))  # [N,d1*d2..,C]-> [N*d1*d2..,C]
        ori_shp = target.shape
        target = target.view(-1, 1)  # [N,d1,d2,...]->[N*d1*d2*...,1]
        valid_mask = None
        if self.ignore_index is not None:
            valid_mask = target != self.ignore_index
            target = target * valid_mask

        # ----------memory saving way--------
        prob = prob.gather(1, target).view(-1) + self.smooth  # avoid nan
        logpt = torch.log(prob)
        # alpha_class = alpha.gather(0, target.view(-1))
        alpha_class = alpha[target.squeeze().long()]
        class_weight = -alpha_class * torch.pow(torch.sub(1.0, prob), self.gamma)
        loss = class_weight * logpt
        if valid_mask is not None:
            loss = loss * valid_mask.squeeze()

        if self.reduction == 'mean':
            loss = loss.mean()
            if valid_mask is not None:
                loss = loss.sum() / valid_mask.sum()
        elif self.reduction == 'none':
            loss = loss.view(ori_shp)
        return loss
if __name__ == '__main__':
    seed_torch()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    ##########    dataset
    # print(len(train_data)) # train_data 总共7528条数据
    train_data,test_data = read_data_from_csv()
    train_text = list(train_data['text'])
    train_label = list(train_data['class'])
    ## 2. 增加低比例样本，随机采样
    # neg_index = []
    # pos_index = []
    # for i in range(len(train_label)):
    #     if train_label[i] == 0:
    #         neg_index.append(i)
    #     elif train_label[i] == 1:
    #         pos_index.append(i)
    # for i in range(50):
    #     pos_rand = pos_index[np.random.randint(len(pos_index))]
    #     train_text.append(train_data.iloc[pos_rand]['text'])
    #     train_label.append(train_data.iloc[pos_rand]['class'])
    #
    #     pos_rand = pos_index[np.random.randint(len(pos_index))]
    #     train_text.append(train_data.iloc[pos_rand]['text'])
    #     train_label.append(train_data.iloc[pos_rand]['class'])
    #
    #     neg_rand = neg_index[np.random.randint(len(neg_index))]
    #     train_text.append(train_data.iloc[neg_rand]['text'])
    #     train_label.append(train_data.iloc[neg_rand]['class'])
    ## 3. 带权重的交叉熵
    # criterion = nn.CrossEntropyLoss()

    # pos = 1  / len([x for x in train_label if x == 1])
    # neg = 1 / len([x for x in train_label if x == 0])
    # neu = 1 / len([x for x in train_label if x == 2])
    # weight = torch.tensor([neg,pos,neu],dtype = torch.float)
    # weight.to(device)
    # print(weight)
    # criterion = nn.CrossEntropyLoss(weight = weight)

    ## 4. Focal Loss
    # criterion = FocalLoss_Ori(num_class=3)
    train_dataset = sentiment_dataset(train_text,train_label)
    # train_dataset,valid_dataset = random_split(train_dataset,[7000,528],generator=torch.Generator().manual_seed(42))
    train_data_loader = DataLoader(
                    train_dataset,
                    batch_size = 64,
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


    model = sentiment_model()
    # 5.  baseline CrossEntropy
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

        print('epoch:%d,total_loss:%.5f,accu:%.5f' % (epoch, train_loss * epoch / global_step,mean_loss))
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