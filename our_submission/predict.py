import pandas as pd
from transformers import BertTokenizer, BertModel,AutoTokenizer, AutoModel
from model import sentiment_model, BiLSTM_CRF,BERT_CRF
import torch
from preprocess import read_data_from_csv, sentiment_dataset,ner_dataset,data_process,bert_ner_dataset
from torch.utils.data import DataLoader
from sentiment_task import sentiment_collate_fn
from utils import ner_accuary2,calculate_kappa
tokenizer = AutoTokenizer.from_pretrained("nghuyong/ernie-1.0")
def predict_sentiment(model_file_name,test_data_loader):
    model = sentiment_model()
    model.load_state_dict(torch.load(model_file_name))
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model.to(device)
    result = []
    with torch.no_grad():
        for i, batch in enumerate(test_data_loader):
            input_ids, token_type_ids, attention_mask, labels = batch
            input_ids = input_ids.to(device, dtype=torch.long)
            token_type_ids = token_type_ids.to(device, dtype=torch.long)
            attention_mask = attention_mask.to(device, dtype=torch.long)
            labels = labels.to(device, dtype=torch.long)
            probs = model(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
            result.extend(torch.argmax(probs,-1).squeeze().cpu().numpy().tolist())
    return result

def bert_test_collate_fn(batch,token_id=tokenizer.pad_token_id,token_type_id=tokenizer.pad_token_type_id):
    inputs_ids,token_type_ids,attention_mask,text_lengths = zip(*batch)
    max_len = max([len(seq_a) for seq_a in inputs_ids]) #这里我使用的是一个batch中text_a或者是text_b的最大长度作为max_len,也可以自定义长度
    inputs_ids = [seq + [token_id]*(max_len-len(seq)) for seq in inputs_ids]
    token_type_ids = [seq + [token_type_id]*(max_len-len(seq)) for seq in token_type_ids]
    attention_mask = [seq + [0]*(max_len-len(seq)) for seq in attention_mask]
    return torch.tensor(inputs_ids,dtype = torch.long),torch.tensor(token_type_ids,dtype = torch.long),torch.tensor(attention_mask,dtype = torch.long),torch.tensor(text_lengths,dtype=torch.long)


def predict_ner(test_data_loader,tag_to_ix):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = BERT_CRF(tag_to_ix)
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
        for pre,length in zip(pred,text_lengths):
            pre = pre[:length]
            ret = []
            for id in pre:
                ret.append(id_to_tag[id])
            BIO.append(' '.join(ret))
    return BIO
if __name__ == '__main__':

    _,test_data = read_data_from_csv()
    ret = pd.DataFrame(columns=['id','text','BIO_anno','class'])
    ret['id'] = list(range(len(test_data)))
    ret['text'] = test_data['text']

    ## predict sentiment

    test_dataset =  sentiment_dataset(test_data['text'],is_train = False)
    test_data_loader = DataLoader(
                    test_dataset,
                    batch_size = 32,
                    collate_fn = sentiment_collate_fn,
                    shuffle=False
    )
    sentiment_result = predict_sentiment('./model/sentiment_best_model.pt',test_data_loader)
    ret['class'] = sentiment_result
    ret.to_csv('our_submission.csv', index=None)


    ## ner
    tag_to_ix = {'B-BANK': 0, 'I-BANK': 1, 'O': 2, 'B-COMMENTS_N': 3, 'I-COMMENTS_N': 4, 'B-COMMENTS_ADJ': 5,
                 'I-COMMENTS_ADJ': 6, 'B-PRODUCT': 7, 'I-PRODUCT': 8, '<START>': 9, '<STOP>': 10}
    ## datasets
    test_datasets = bert_ner_dataset(test_data['text'], tag_to_ix,is_train=False)
    test_data_loader = DataLoader(
        test_datasets,
        batch_size=32,
        collate_fn=bert_test_collate_fn,
        shuffle=False
    )
    ## train parameters
    BIO = predict_ner(test_data_loader,tag_to_ix)
    ret['BIO_anno'] = BIO
    ret.to_csv('MG21370009.csv', index=None)

    ## cacluate F1, kappa

    # _,_,F1 = ner_accuary2(list(ret['BIO_anno']),list(test_data['BIO_anno']))
    # kappa = calculate_kappa(ret,test_data)
    # score = 0.5 * F1 + 0.5 * kappa
    # print("F1:%.5f,kappa:%.5f,score:%.5f" %(F1,kappa,score))






