import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from transformers import  BertModel,BertTokenizer,AutoTokenizer, AutoModel
from trainer import BERT_TRAINER,BERT_CRF_TRAINER,BiLSTM_CRF_TRAINER,BERT_CRF2_TRAINER
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
import torch
import os
from utils import load_vectors,ner_accuary
from preprocess import data_process,read_data_from_csv,bert_ner_dataset
from model import BERT_CRF

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
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    seed_torch()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    train_data, test_data = read_data_from_csv()
    tokenizer = AutoTokenizer.from_pretrained("nghuyong/ernie-1.0")
    tag_to_ix = {'B-BANK': 0, 'I-BANK': 1, 'O': 2, 'B-COMMENTS_N': 3, 'I-COMMENTS_N': 4, 'B-COMMENTS_ADJ': 5,
                 'I-COMMENTS_ADJ': 6, 'B-PRODUCT': 7, 'I-PRODUCT': 8, '<START>': 9, '<STOP>': 10}
    ## train
    bilstm_trainer = BiLSTM_CRF_TRAINER()
    # bert_trainer = BERT_TRAINER()
    # bert_crf2_trainer = BERT_CRF2_TRAINER()
    # bert_crf_trainer = BERT_CRF_TRAINER()
    # bert_trainer.train()
    bilstm_trainer.train()