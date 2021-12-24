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
from preprocess import data_process,read_data_from_csv,lstm_ner_dataset,bert_ner_dataset
class sentiment_model(nn.Module):
    def __init__(self,dropout=None):
        super(sentiment_model,self).__init__()
        self.bertmodel = AutoModel.from_pretrained("nghuyong/ernie-1.0")
        self.dropout = nn.Dropout(dropout if dropout is not None else 0.2)
        # 情感分类任务，0，1，2
        self.classifier = nn.Linear(self.bertmodel.config.hidden_size, 3)
    def forward(self,input_ids,token_type_ids,attention_mask):
        outputs= self.bertmodel(input_ids,token_type_ids=token_type_ids,attention_mask=attention_mask)#
        outputs = outputs.pooler_output
        outputs = self.dropout(outputs)
        #
        logits = self.classifier(outputs)
        return logits

class BERT(nn.Module):  ## 直接用BERT + MLP 做NER
    def __init__(self, tag_to_ix):
        super(BERT, self).__init__()
        self.tag_to_ix = tag_to_ix
        self.bert = AutoModel.from_pretrained("nghuyong/ernie-1.0")
        self.target_size = len(tag_to_ix)
        self.dropout = nn.Dropout(0.2)
        # NER
        self.classifier = nn.Linear(self.bert.config.hidden_size, self.target_size)

    def forward(self, inputs_ids, token_type_ids, attention_mask):
        feats = self.bert(inputs_ids, token_type_ids, attention_mask)
        outputs = self.dropout(feats.last_hidden_state)
        outputs = self.classifier(outputs)
        return outputs # 返回logits再送入cross_entropy即可
class BERT_CRF(nn.Module):
    def __init__(self,tag_to_ix):
        super(BERT_CRF,self).__init__()
        self.tag_to_ix = tag_to_ix
        self.crf = CRF(self.tag_to_ix)
        self.bert = AutoModel.from_pretrained("nghuyong/ernie-1.0")
        self.target_size = len(tag_to_ix)
        self.dropout = nn.Dropout(0.2)
        # NER
        self.classifier = nn.Linear(self.bert.config.hidden_size, self.target_size)

    def forward(self,inputs_ids,token_type_ids,attention_mask,label,text_lengths):
        feats = self.bert(inputs_ids,token_type_ids,attention_mask)
        outputs = self.dropout(feats.last_hidden_state)
        outputs = self.classifier(outputs)
        # crf
        loss = self.crf.neg_log_likelihood(outputs[:,1:,:], label[:,1:],text_lengths) # 去除第一个CLS label和输出
        return loss
    def decode(self,inputs_ids,token_type_ids,attention_mask,text_lengths):
        bert_feats = self.bert(inputs_ids,token_type_ids,attention_mask)
        outputs = self.dropout(bert_feats.last_hidden_state)
        outputs = self.classifier(outputs)
        paths = []
        for feats,length in zip(outputs,text_lengths): ##text_length是不含padding的数据
            feats = feats[1:length+1] ##去除开头的cls和后面的padding
            path = self.crf.decode(feats)
            paths.append(path)
        return paths


class BERT_CRF2(nn.Module):
    def __init__(self,tag_to_ix):
        super(BERT_CRF2,self).__init__()
        self.tag_to_ix = tag_to_ix
        self.crf = CRF2(self.tag_to_ix)
        self.bert = AutoModel.from_pretrained("nghuyong/ernie-1.0")
        self.target_size = len(tag_to_ix)
        self.dropout = nn.Dropout(0.2)
        # NER
        self.classifier = nn.Linear(self.bert.config.hidden_size, self.target_size)

    def forward(self,inputs_ids,token_type_ids,attention_mask,label,text_lengths):
        feats = self.bert(inputs_ids,token_type_ids,attention_mask)
        outputs = self.dropout(feats.last_hidden_state)
        outputs = self.classifier(outputs)
        # crf
        loss = self.crf.neg_log_likelihood(outputs[:,1:,:], label[:,1:],text_lengths) # 去除第一个CLS label和输出
        return loss
    def decode(self,inputs_ids,token_type_ids,attention_mask,text_lengths):
        bert_feats = self.bert(inputs_ids,token_type_ids,attention_mask)
        outputs = self.dropout(bert_feats.last_hidden_state)
        outputs = self.classifier(outputs)
        paths = []
        for feats,length in zip(outputs,text_lengths): ##text_length是不含padding的数据
            feats = feats[1:length+1] ##去除开头的cls和后面的padding
            path = self.crf.decode(feats)
            paths.append(path)
        return paths
class BiLSTM_CRF(nn.Module):
    def __init__(self,vocab_size,tag_to_ix,embedding,hidden_dim):
        super(BiLSTM_CRF,self).__init__()
        self.vocab_size = vocab_size
        self.tag_to_ix = tag_to_ix
        self.embedding = embedding
        self.hidden_dim = hidden_dim
        self.crf = CRF(self.tag_to_ix)
        self.lstm = LSTM(self.vocab_size,self.tag_to_ix,self.embedding,self.hidden_dim)
    def forward(self,data,label,text_lengths):
        feats = self.lstm(data)
        loss = self.crf.neg_log_likelihood(feats, label,text_lengths)
        return loss
    def decode(self,data,text_lengths,tag_to_ix):
        batch_feats = self.lstm(data)
        paths = []
        for feats,length in zip(batch_feats,text_lengths):
            feats = feats[:length]
            path = self.crf.decode(feats)
            paths.append(path)

        return paths


class LSTM(nn.Module):
    def __init__(self, vocab_size, tag_to_ix, embedding, hidden_dim):
        super(LSTM, self).__init__()
        self.START_TAG = "<START>"
        self.STOP_TAG = "<STOP>"
        self.embedding_dim = len(embedding[0])
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.tag_to_ix = tag_to_ix
        self.target_size = len(tag_to_ix)
        self.word_embeds = nn.Embedding.from_pretrained(embedding)
        self.lstm = nn.LSTM(self.embedding_dim, hidden_dim // 2,
                            num_layers=1, bidirectional=True,batch_first = True)

        self.hidden2tag = nn.Linear(hidden_dim, self.target_size)

        self.hidden = self.init_hidden()
    def init_hidden(self):
        return (torch.randn(2, 1, self.hidden_dim // 2),
                torch.randn(2, 1, self.hidden_dim // 2))
    def forward(self,data):
        embeds = self.word_embeds(data)
        packed_output, self.hidden = self.lstm(embeds)
        output = self.hidden2tag(packed_output)
        return output

class CRF(nn.Module):
    def __init__(self,tag_to_ix):
        super(CRF,self).__init__()
        self.tag_to_ix = tag_to_ix
        self.target_size = len(tag_to_ix)
        self.transitions = nn.Parameter(
            torch.randn(self.target_size, self.target_size))
        self.START_TAG = "<START>"
        self.STOP_TAG = "<STOP>"
        # These two statements enforce the constraint that we never transfer
        # to the start tag and we never transfer from the stop tag
        self.transitions.data[:,self.tag_to_ix[self.START_TAG]] = -10000  # 从行转到列
        self.transitions.data[self.tag_to_ix[self.STOP_TAG], :] = -10000
    def forward(self,feats):
        previous = None
        for i,feat in enumerate(feats):
            if i == 0:
                previous = feat  + self.transitions.data[self.tag_to_ix[self.START_TAG],:]                   ##  START_TAG到第一个状态的转移概率
            else:
                previous = previous.view(1, -1).expand(self.target_size, self.target_size).transpose(1, 0)
                feat = feat.view(1,-1).expand(self.target_size,self.target_size)
                score = previous + feat + self.transitions.data
                max_score = torch.max(score,dim=0).values # 每一列的最大值
                # print("previous:",previous)
                # print("feat:",feat)
                # print("score:",score)
                # print("max_score:",max_score)
                max_score_broadcast = max_score.view(1,-1).expand(self.target_size,self.target_size) ## 防止结果是inf
                # print("max_score_broadcast:",max_score_broadcast)
                score_exp = torch.exp(score - max_score_broadcast)
                # print("score_exp:",score_exp)
                previous = max_score + torch.log(torch.sum(score_exp,dim=0))
                # Compute log sum exp in a numerically stable way for the forward algorithm
                # 前向算法是不断累积之前的结果，这样就会有个缺点
                # 指数和累积到一定程度后，会超过计算机浮点值的最大值，变成inf，这样取log后也是inf
                # 为了避免这种情况，用一个合适的值clip去提指数和的公因子，这样就不会使某项变得过大而无法计算
                # SUM = log(exp(s1)+exp(s2)+...+exp(s100))
                #     = log{exp(clip)*[exp(s1-clip)+exp(s2-clip)+...+exp(s100-clip)]}
                #     = clip + log[exp(s1-clip)+exp(s2-clip)+...+exp(s100-clip)]
                # where clip=max

        max_previous = torch.max(previous,dim=0).values ## 防止结果是inf
        # print('max_previous:',max_previous)
        forward_score =  max_previous +  torch.log(torch.sum(torch.exp(previous - max_previous.view(1,-1).expand(1,self.target_size))))
        # print(torch.exp(previous))
        # print(torch.sum(torch.exp(previous)))
        return forward_score
    def gold_score(self,feats, tags):
        score = 0
        for i,(feat,tag) in enumerate(zip(feats,tags)):
            score += feat[tag]
            if i == 0:
                score += self.transitions.data[self.tag_to_ix[self.START_TAG]][tag]  ##  START_TAG到第一个状态的转移概率
            else:
                if self.transitions.data[tags[i-1]][tag] == -10000.:
                    pass
                    # print('feats',feats)
                    # print('tags',tags)
                    # print(self.transitions.data)
                score += self.transitions.data[tags[i-1]][tag]
        score += self.transitions.data[tags[i]][self.tag_to_ix[self.STOP_TAG]]   ##  最后一个状态到STOP_TAG的转移概率
        return score
    def neg_log_likelihood(self,feats,label,text_lengths):
        loss = 0
        for feat,lab,length in zip(feats,label,text_lengths):
            forward_score = self.forward(feat[:length])
            gold_score = self.gold_score(feat[:length],lab[:length])
            loss += forward_score - gold_score
        return loss / feats.size(0)
    # def decode(self,feats):
    #     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #     viterbi_score = torch.tensor(torch.full((1, self.target_size), -10000.),device = device)
    #     viterbi_score[0][self.tag_to_ix[self.START_TAG]] = 0
    #     backpointers = []
    #     for i,feat in enumerate(feats):
    #         each_back = []
    #         each_viterbi = []
    #         for j in range(self.target_size):
    #             cur_score = viterbi_score + feat + self.transitions.data[:,j]  ##  算了START_TAG到第一个状态的转移概率
    #             max_element = torch.max(cur_score,-1)
    #             each_back.append(max_element.indices)
    #             each_viterbi.append(max_element.values)
    #         backpointers.append(each_back)
    #         viterbi_score = torch.tensor(each_viterbi,device=device)
    #     viterbi_score += self.transitions.data[:,self.tag_to_ix[self.STOP_TAG]] ##  算了最后一个状态到STOP_TAG的转移概率
    #
    #     best_tag_id = torch.max(viterbi_score,0).indices
    #     best_path = [best_tag_id]
    #     for bptrs_t in reversed(backpointers):
    #         best_tag_id = bptrs_t[best_tag_id]
    #         best_path.append(best_tag_id)
    #
    #     start = best_path.pop()
    #     assert start == self.tag_to_ix[self.START_TAG]  # Sanity check
    #     best_path.reverse()
    #     return best_path
    def decode(self, feats):
        backpointers = []
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # 初始化viterbi的previous变量
        init_vvars = torch.tensor(torch.full((1, self.target_size), -10000.),device=device)
        init_vvars[0][self.tag_to_ix[self.START_TAG]] = 0
        previous = init_vvars
        for obs in feats:
            # 保存当前时间步的回溯指针
            bptrs_t = []
            # 保存当前时间步的viterbi变量
            viterbivars_t = []

            for next_tag in range(self.target_size):
                # 维特比算法记录最优路径时只考虑上一步的分数以及上一步tag转移到当前tag的转移分数(因为加了stop_tag，可以倒推）
                # 并不取决与当前tag的发射分数
                next_tag_var = previous + self.transitions[:,next_tag]
                best_tag_id = argmax(next_tag_var)
                bptrs_t.append(best_tag_id)
                viterbivars_t.append(next_tag_var[0][best_tag_id].view(1))
            # 更新previous，加上当前tag的发射分数obs
            previous = (torch.cat(viterbivars_t) + obs).view(1, -1)
            # 回溯指针记录当前时间步各个tag来源前一步的tag
            backpointers.append(bptrs_t)

        # Transition to STOP_TAG
        # 考虑转移到STOP_TAG的转移分数
        terminal_var = previous + self.transitions[:,self.tag_to_ix[self.STOP_TAG]]
        best_tag_id = argmax(terminal_var)
        path_score = terminal_var[0][best_tag_id]

        # 通过回溯指针解码出最优路径
        best_path = [best_tag_id]
        # best_tag_id作为线头，反向遍历backpointers找到最优路径
        for bptrs_t in reversed(backpointers):
            best_tag_id = bptrs_t[best_tag_id]
            best_path.append(best_tag_id)
        # 去除START_TAG
        start = best_path.pop()
        assert start == self.tag_to_ix[self.START_TAG]  # Sanity check
        best_path.reverse()
        return  best_path
class CRF2(nn.Module):
    def __init__(self,tag_to_ix):
        super(CRF2, self).__init__()
        self.tag_to_ix = tag_to_ix
        self.target_size = len(tag_to_ix)
        self.transitions = nn.Parameter(
            torch.randn(self.target_size, self.target_size))
        self.START_TAG = "<START>"
        self.STOP_TAG = "<STOP>"
        # These two statements enforce the constraint that we never transfer
        # to the start tag and we never transfer from the stop tag
        self.transitions.data[self.tag_to_ix[self.START_TAG],:] = -10000  # 从列转到行
        self.transitions.data[:,self.tag_to_ix[self.STOP_TAG]] = -10000

    def _score_sentence(self, feats, tags):
        # 计算给定tag序列的分数，即一条路径的分数
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        score = torch.tensor(torch.zeros(1),device=device)
        tags = torch.cat([torch.tensor([self.tag_to_ix[self.START_TAG]], dtype=torch.long,device = device), tags])
        for i, feat in enumerate(feats):
            # 递推计算路径分数：转移分数 + 发射分数
            score = score + self.transitions[tags[i + 1], tags[i]] + feat[tags[i + 1]]
        score = score + self.transitions[self.tag_to_ix[self.STOP_TAG], tags[-1]]
        return score

    def _forward_alg(self, feats):
        # 通过前向算法递推计算
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        init_alphas = torch.tensor(torch.full((1, self.target_size), -10000.),device=device)
        # 初始化step 0即START位置的发射分数，START_TAG取0其他位置取-10000
        init_alphas[0][self.tag_to_ix[self.START_TAG]] = 0.
        # 将初始化START位置为0的发射分数赋值给previous
        previous = init_alphas
        # 迭代整个句子
        for obs in feats:
            # 当前时间步的前向tensor
            alphas_t = []
            for next_tag in range(self.target_size):
                # 取出当前tag的发射分数，与之前时间步的tag无关
                emit_score = obs[next_tag].view(1, -1).expand(1, self.target_size)
                # 取出当前tag由之前tag转移过来的转移分数
                trans_score = self.transitions[next_tag].view(1, -1)
                # 当前路径的分数：之前时间步分数 + 转移分数 + 发射分数
                next_tag_var = previous + trans_score + emit_score
                # 对当前分数取log-sum-exp
                alphas_t.append(log_sum_exp(next_tag_var).view(1))
            # 更新previous 递推计算下一个时间步
            previous = torch.cat(alphas_t).view(1, -1)
        # 考虑最终转移到STOP_TAG
        terminal_var = previous + self.transitions[self.tag_to_ix[self.STOP_TAG]]
        # 计算最终的分数
        scores = log_sum_exp(terminal_var)
        return scores


    def _viterbi_decode(self, feats):
        backpointers = []
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # 初始化viterbi的previous变量
        init_vvars = torch.tensor(torch.full((1, self.target_size), -10000.),device=device)
        init_vvars[0][self.tag_to_ix[self.START_TAG]] = 0
        previous = init_vvars
        for obs in feats:
            # 保存当前时间步的回溯指针
            bptrs_t = []
            # 保存当前时间步的viterbi变量
            viterbivars_t = []

            for next_tag in range(self.target_size):
                # 维特比算法记录最优路径时只考虑上一步的分数以及上一步tag转移到当前tag的转移分数
                # 并不取决与当前tag的发射分数
                next_tag_var = previous + self.transitions[next_tag]
                best_tag_id = argmax(next_tag_var)
                bptrs_t.append(best_tag_id)
                viterbivars_t.append(next_tag_var[0][best_tag_id].view(1))
            # 更新previous，加上当前tag的发射分数obs
            previous = (torch.cat(viterbivars_t) + obs).view(1, -1)
            # 回溯指针记录当前时间步各个tag来源前一步的tag
            backpointers.append(bptrs_t)

        # Transition to STOP_TAG
        # 考虑转移到STOP_TAG的转移分数
        terminal_var = previous + self.transitions[self.tag_to_ix[self.STOP_TAG]]
        best_tag_id = argmax(terminal_var)
        path_score = terminal_var[0][best_tag_id]

        # 通过回溯指针解码出最优路径
        best_path = [best_tag_id]
        # best_tag_id作为线头，反向遍历backpointers找到最优路径
        for bptrs_t in reversed(backpointers):
            best_tag_id = bptrs_t[best_tag_id]
            best_path.append(best_tag_id)
        # 去除START_TAG
        start = best_path.pop()
        assert start == self.tag_to_ix[self.START_TAG]  # Sanity check
        best_path.reverse()
        return path_score, best_path

    def neg_log_likelihood(self,feats,label,text_lengths):
        loss = 0
        for feat,lab,length in zip(feats,label,text_lengths):
            forward_score = self._forward_alg(feat[:length])
            gold_score = self._score_sentence(feat[:length],lab[:length])
            loss += forward_score - gold_score
        return loss / feats.size(0)
        return forward_score - gold_score

    def decode(self, feats):
        # 根据发射分数以及转移分数，通过viterbi解码找到一条最优路径
        _, tag_seq = self._viterbi_decode(feats)
        return tag_seq

## CRF2的工具函数
def argmax(vec):
    # return the argmax as a python int
    # 返回vec的dim为1维度上的最大值索引
    _, idx = torch.max(vec, 1)
    return idx.item()

def prepare_sequence(seq, to_ix):
    # 将句子转化为ID
    idxs = [to_ix[w] for w in seq]
    return torch.tensor(idxs, dtype=torch.long)


# Compute log sum exp in a numerically stable way for the forward algorithm
# 前向算法是不断累积之前的结果，这样就会有个缺点
# 指数和累积到一定程度后，会超过计算机浮点值的最大值，变成inf，这样取log后也是inf
# 为了避免这种情况，用一个合适的值clip去提指数和的公因子，这样就不会使某项变得过大而无法计算
# SUM = log(exp(s1)+exp(s2)+...+exp(s100))
#     = log{exp(clip)*[exp(s1-clip)+exp(s2-clip)+...+exp(s100-clip)]}
#     = clip + log[exp(s1-clip)+exp(s2-clip)+...+exp(s100-clip)]
# where clip=max
def log_sum_exp(vec):
    max_score = vec[0, argmax(vec)]
    max_score_broadcast = max_score.view(1, -1).expand(1, vec.size()[1])
    return max_score + torch.log(torch.sum(torch.exp(vec - max_score_broadcast)))




if __name__ == '__main__':
    pass
    # test BiLSTM_CRF
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # fast_text,word2idx = load_vectors("../../ff/cc.zh.300.vec")  # fast_text词向量位置
    # train_data,test_data = read_data_from_csv()
    # ## datasets
    # input_ids, label_ids, tag_to_ix, text_lengths = data_process(train_data['text'], train_data['BIO_anno'],word2idx = word2idx)
    # vocab_size = len(word2idx)
    # embedding = torch.tensor(fast_text,dtype=torch.float)
    # hidden_dim = 100
    # model = BiLSTM_CRF(vocab_size, tag_to_ix, embedding, hidden_dim)
    # optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    #
    # train_datasets = lstm_ner_dataset(input_ids,label_ids,text_lengths)
    # train_datasets, valid_datasets = torch.utils.data.random_split(train_datasets, [5522, 500],
    #                                                                generator=torch.Generator().manual_seed(42))
    # train_data_loader = DataLoader(train_datasets, batch_size=32, num_workers=4,collate_fn=lstm_ner_collate_fn, shuffle=True)
    # valid_data_loader = DataLoader(valid_datasets, batch_size=32, num_workers=4,collate_fn=lstm_ner_collate_fn, shuffle=True)
    # model.to(device)
    # model.train()
    # epochs= 5
    # step = 0
    # for epoch in range(epochs):
    #     model.train()
    #     total_F1 = []
    #     total_loss = []
    #     for batch in train_data_loader:
    #         optimizer.zero_grad()
    #         X,label,text_lengths = batch
    #         X = X.to(device)
    #         label = label.to(device)
    #         text_lengths = text_lengths.to(device)
    #         loss = model.forward(X,label,text_lengths)
    #         total_loss.append(loss)
    #         ret = model.decode(X,text_lengths,tag_to_ix)
    #         print('epoch:%d,step:%d,loss:%.5f' % (epoch,step,loss))
    #         step += 1
    #         loss.backward()
    #         nn.utils.clip_grad_norm_(model.parameters(), max_norm=20, norm_type=2)
    #         optimizer.step()
    #     print("epoch:%d,mean_loss : %.5f" % (epoch, sum(total_loss) / len(total_loss)))
    #     model.eval()
    #     for batch in valid_data_loader:
    #         X,labels,text_lengths = batch
    #         X = X.to(device)
    #         label = label.to(device)
    #         text_lengths = text_lengths.to(device)
    #         pred = model.decode(X, text_lengths, tag_to_ix)
    #         p, r, F1 = ner_accuary(pred, labels, tag_to_ix, text_lengths)
    #         total_F1.append(F1)
    #         tags = []
    #         ret = []
    #         id_to_tag = list(tag_to_ix.keys())
    #         # 展示每个batch的第一个数据结果
    #         for id in labels[0][text_lengths[0]]:
    #             tags.append(id_to_tag[id])
    #         for id in pred[0][:text_lengths[0]]:
    #             ret.append(id_to_tag[id])
    #         print('orgin-tag:', tags)
    #         print('predict-tag:', ret)
    #         print("F1 score:", F1)
    #     cur_F1 = sum(total_F1) / len(total_F1)
    #     print("valid F1 score:", cur_F1)
    #     if cur_F1 > valid_F1:
    #         valid_F1 = cur_F1


