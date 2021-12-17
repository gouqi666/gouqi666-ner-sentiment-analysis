from transformers import BertTokenizer, BertModel,AutoTokenizer, AutoModel
from torch.utils.data import Dataset
import pandas as pd
from gensim.models.keyedvectors import FastTextKeyedVectors
# import torchtext.vocab as vocab
import torch

def read_data_from_csv():
    train_data = pd.read_csv('./train.csv')
    test_data = pd.read_csv('./test.csv')

    train_data['BIO_anno'] = train_data['BIO_anno'].apply(lambda x:x.split(' '))
    train_data['training_data'] = train_data.apply(lambda row: (list(row['text']), row['BIO_anno']), axis = 1)
    test_data['testing_data'] = test_data.apply(lambda row: (list(row['text'])), axis = 1)
    return train_data,test_data


def data_process(data, label=None,word2idx = None, is_train = True):  # 这里直接按照空格进行分词了
    START_TAG = "<START>"
    STOP_TAG = "<STOP>"
    def _process_data(sentences,word2idx):
        input_ids = []
        text_lengths = []
        for sen in sentences:  # sen
            sen_ids = []
            for word in sen:
                if word in word2idx:
                    sen_ids.append(word2idx[word])
                else:
                    sen_ids.append(word2idx['unk'])
            input_ids.append(sen_ids)
            text_lengths.append(len(sen_ids))
        return input_ids, text_lengths

    def _process_label(label):
        lab2idx = {}
        idx2lab = []
        label_ids = []
        for sen in label:
            sen_lab_ids = []
            for tag in sen:
                if tag in lab2idx:
                    sen_lab_ids.append(lab2idx[tag])
                else:
                    lab2idx[tag] = len(idx2lab)
                    idx2lab.append(tag)
                    sen_lab_ids.append(lab2idx[tag])
            label_ids.append(sen_lab_ids)
        lab2idx[START_TAG] = len(lab2idx)
        lab2idx[STOP_TAG] = len(lab2idx)
        return label_ids, lab2idx

    ##   lstm_crf,使用nltk tokenizer和glove词向量
    input_ids, text_lengths = _process_data(data,word2idx)
    if is_train == False:
        return input_ids, text_lengths
    label_ids, lab2idx = _process_label(label)
    return input_ids, label_ids, lab2idx, text_lengths

class bert_ner_dataset(Dataset):
    def __init__(self,data,label = None,tag_to_ix = None,is_train =True):
        super(bert_ner_dataset,self).__init__()
        self.tokenizer = AutoTokenizer.from_pretrained("nghuyong/ernie-1.0")
        self.input_ids = []
        self.token_type_ids = []
        self.attention_mask = []
        self.text_lengths = []
        self.label = label
        self.is_train = is_train
        if self.is_train:
            ret = []
            for tags in label:
                ret.append([tag_to_ix[tag] for tag in tags])
            self.label = ret
        for text in data:
            x,y,z = [101],[0],[1]
            for word in text:
                encoder_inputs = self.tokenizer(word)
                x.append(encoder_inputs['input_ids'][1])
                y.append(encoder_inputs['token_type_ids'][1])
                z.append(encoder_inputs['attention_mask'][1])
            x.append(102)
            y.append(0)
            z.append(1)
            self.input_ids.append(x)
            self.token_type_ids.append(y)
            self.attention_mask.append(z)
            self.text_lengths.append(len(text))

    def __getitem__(self, idx):
        input_ids = self.input_ids[idx]
        token_type_ids = self.token_type_ids[idx]
        attention_mask = self.attention_mask[idx]
        text_lengths = self.text_lengths[idx]
        if self.is_train:
            return input_ids, token_type_ids, attention_mask,self.label[idx],text_lengths
        else:
            return input_ids, token_type_ids, attention_mask,text_lengths

    def __len__(self):
        return len(self.input_ids)
class ner_dataset(Dataset):
    def __init__(self,data,label=None,text_lengths = None,is_train = True):
        super(ner_dataset).__init__()
        self.input_ids = data
        self.label = label
        self.text_lengths = text_lengths
        self.is_train = is_train
    def __getitem__(self,idx):
        input_ids = self.input_ids[idx]
        if self.is_train:
            return input_ids,self.label[idx],self.text_lengths[idx]
        else:
            return input_ids,self.text_lengths[idx]
    def __len__(self):
        return len(self.input_ids)

# class ner_dataset(Dataset):
#     def __init__(self,data,label=None,is_train = True):
#         super(ner_dataset).__init__()
#         self.tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
#         self.input_ids = []
#         self.token_type_ids = []
#         self.attention_mask = []
#         self.label = label
#         self.is_train = is_train
#         for text in data:
#             encoder_inputs = self.tokenizer(text)
#             self.input_ids.append(encoder_inputs['input_ids'])
#             self.token_type_ids.append(encoder_inputs['token_type_ids'])
#             self.attention_mask.append(encoder_inputs['attention_mask'])
#     def __getitem__(self,idx):
#         input_ids = self.input_ids[idx]
#         token_type_ids = self.token_type_ids[idx]
#         attention_mask = self.attention_mask[idx]
#         label = self.label
#         if self.is_train:
#             return input_ids,token_type_ids,attention_mask,label[idx]
#         else:
#             return input_ids, token_type_ids,attention_mask,[1]
#     def __len__(self):
#         return len(self.input_ids)

class sentiment_dataset(Dataset):
    def __init__(self,data,label=None,is_train = True):
        super(sentiment_dataset).__init__()
        self.tokenizer = AutoTokenizer.from_pretrained("nghuyong/ernie-1.0")
        self.input_ids = []
        self.token_type_ids = []
        self.attention_mask = []
        self.label = label
        self.is_train = is_train
        for text in data:
            encoder_inputs = self.tokenizer(text)
            self.input_ids.append(encoder_inputs['input_ids'])
            self.token_type_ids.append(encoder_inputs['token_type_ids'])
            self.attention_mask.append(encoder_inputs['attention_mask'])
    def __getitem__(self,idx):
        input_ids = self.input_ids[idx]
        token_type_ids = self.token_type_ids[idx]
        attention_mask = self.attention_mask[idx]
        label = self.label
        if self.is_train:
            return input_ids,token_type_ids,attention_mask,label[idx]
        else:
            return input_ids, token_type_ids,attention_mask,[1]
    def __len__(self):
        return len(self.input_ids)

