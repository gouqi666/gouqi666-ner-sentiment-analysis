import pandas as pd
total_data = pd.read_csv('./our_submission/train_data_public.csv')
test_data = pd.read_csv('./our_submission/test.csv')
print(test_data.info())
count = 0
for x in test_data['text']:
    if x in list(total_data['text']):
        count += 1
    else:
        print(x)
print(count)
ret = pd.merge(test_data,total_data,on="text",how='inner')
del ret['id_y']
ret['id'] = ret['id_x']
del ret['id_x']
ret.to_csv('./our_submission/test.csv',index=None)