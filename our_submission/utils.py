import torch
import io
def load_vectors(fname):
    fin = io.open(fname, 'r', encoding='utf8', newline='\n', errors='ignore')
    n, d = map(int, fin.readline().split())
    data = []
    word2idx = {}
    idx = 0
    for line in fin:
        tokens = line.rstrip().split(' ')
        vector = list(map(float, tokens[1:]))
        if len(vector) != 300:
            continue
        data.append(vector)
        if tokens[0] not in word2idx:
            word2idx[tokens[0]] = idx
            idx += 1
    data.append([0] * 300)
    word2idx['unk'] = idx
    return data,word2idx
def ner_accuary(seq,label,lab2idx,text_lengths): ## label pad后的数据
    gold_num = 0
    predict_num = 0
    correct_num = 0
    idx2lab = list(lab2idx.keys())
    for each_seq,each_lab,length in zip(seq,label,text_lengths):
        flag =False
        each_lab = each_lab[:length]
        for pre,tag in zip(each_seq,each_lab):
            if idx2lab[pre][0] == 'B':
                predict_num += 1
            if idx2lab[tag][0] == 'B':
                gold_num += 1
            if flag: # 进入配对模式
                if pre == tag:
                    if idx2lab[tag][0] == 'B' or idx2lab[tag][0] == 'O': # 在配对模式下，当前是B，O，说明之前的满足了，需要更新
                        correct_num += 1
                        flag = False
                else: ## pre != tag
                    if  idx2lab[tag][0] != 'I':
                        correct_num += 1  ##之前的配对成功
                    flag = False
            if not flag  and idx2lab[tag][0] == 'B' and pre == tag:
                flag = True
        if flag:  ## 最后一个配对满足
            correct_num += 1

    if predict_num == 0:
        precise = 0
    else:
        precise = correct_num / predict_num
    if gold_num == 0:
        recall = 0
    else:
        recall = correct_num / gold_num
    if precise == 0 and recall == 0:
        F1 = 0
    else:
        F1 = 2 * precise * recall / (precise + recall)
    return precise, recall, F1
def ner_accuary2(seq,label): ##
    gold_num = 0
    predict_num = 0
    correct_num = 0
    for each_seq,each_lab in zip(seq,label):
        flag =False
        for pre,tag in zip(each_seq,each_lab):
            if pre[0] == 'B':
                predict_num += 1
            if tag[0] == 'B':
                gold_num += 1
            if flag: # 进入配对模式
                if pre == tag:
                    if tag[0] == 'B' or tag[0] == 'O': # 在配对模式下，当前是B，O，说明之前的满足了，需要更新
                        correct_num += 1
                        flag = False
                else: ## pre != tag
                    if  tag[0] != 'I':
                        correct_num += 1  ##之前的配对成功
                    flag = False
            if not flag  and tag[0] == 'B' and pre == tag:
                flag = True
        if flag:  ## 最后一个配对满足
            correct_num += 1

    if predict_num == 0:
        precise = 0
    else:
        precise = correct_num / predict_num
    if gold_num == 0:
        recall = 0
    else:
        recall = correct_num / gold_num
    if precise == 0 and recall == 0:
        F1 = 0
    else:
        F1 = 2 * precise * recall / (precise + recall)
    return precise, recall, F1
def calculate_kappa(data,label):  ## dataframe
    a0 = (label['class'] == 0).sum() ##真实值
    a1 = (label['class'] == 1).sum()
    a2 = (label['class'] == 2).sum()
    b0 = (data['class'] == 0).sum()
    b1 = (data['class'] == 1).sum()
    b2 = (data['class'] == 2).sum()
    p0 = (label['class'] == data['class']).sum() / len(data)
    pe = (a0 * b0 + a1 * b1 + a2 * b2) / (len(data) ** 2)
    kappa = (p0 - pe) /(1 - pe)
    return kappa
if __name__ == "__main__":
    seq = [[0,1,2,2,0,3,0]] # ['O','B-BANK','I-BANK','I-BANK','O','B-PROUDCT','O']
    label = [[0,1,3,0,0,3,0]] # ['O','B-BANK','B-PROUDCT','O','O','B-PRODUCT','O']
    lab2idx = {'O':0,'B-BANK':1,'I-BANK':2,'B-PRODUCT':3}
    ner_accuary(seq,label,lab2idx)
