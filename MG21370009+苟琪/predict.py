import pandas as pd
from transformers import BertTokenizer, BertModel,AutoTokenizer, AutoModel
import torch
from model import sentiment_model
from trainer import BERT_TRAINER,BERT_CRF_TRAINER,BiLSTM_CRF_TRAINER,BERT_CRF2_TRAINER
from preprocess import read_data_from_csv,sentiment_dataset
from torch.utils.data import DataLoader
from sentiment_task import sentiment_collate_fn
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




if __name__ == '__main__':
    #
    _,test_data = read_data_from_csv()
    ret = pd.DataFrame(columns=['id','text','BIO_anno','class'])
    ret['id'] = list(range(len(test_data)))
    ret['text'] = test_data['text']

    # ## predict sentiment
    #
    test_dataset =  sentiment_dataset(test_data['text'],is_train = False)
    test_data_loader = DataLoader(
                    test_dataset,
                    batch_size = 64,
                    collate_fn = sentiment_collate_fn,
                    shuffle=False
    )
    sentiment_result = predict_sentiment('./model/sentiment_best_model.pt',test_data_loader)
    ret['class'] = sentiment_result
    ret.to_csv('our_submission.csv', index=None)
    #
    #
    # ## ner

    # bilstm_trainer = BiLSTM_CRF_TRAINER()
    # bert_trainer = BERT_TRAINER()
    # bert_crf2_trainer = BERT_CRF2_TRAINER()
    bert_crf_trainer = BERT_CRF_TRAINER()
    # ## train parameters
    # BIO = bilstm_trainer.predict()
    # BIO = bert_trainer.predict()
    BIO = bert_crf_trainer.predict()
    ret['BIO_anno'] = BIO
    ret.to_csv('MG21370009.csv', index=None)
    #







    ## For training data, do cacluate F1, kappa

    # _,_,F1 = ner_accuary2(list(ret['BIO_anno']),list(test_data['BIO_anno']))
    # kappa = calculate_kappa(ret,test_data)
    # score = 0.5 * F1 + 0.5 * kappa
    # print("F1:%.5f,kappa:%.5f,score:%.5f" %(F1,kappa,score))






