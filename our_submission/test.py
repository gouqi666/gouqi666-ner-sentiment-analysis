from preprocess import sentiment_dataset,read_data_from_csv

train_data, test_data = read_data_from_csv()
print(sum(train_data['class'] == 0))
print(sum(train_data['class'] == 1))
print(sum(train_data['class'] == 2)) # 2:0:1 (中立，负面，正面)= 5156:636:230 = 22.4：2.7：1