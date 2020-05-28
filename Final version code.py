# -*- coding: utf-8 -*-

from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer

#1:define a function to remove the specific symbols
import re
def rm_htmltags(text):
    re_tag = re.compile(r'<[^>]+>') 
    return re_tag.sub('', text)  

#2:Load data and emerge content from each class into a list
import os
def read_files(filetype):
    path = "E:/科大学习/2020-2021 spring/1.周一 6010s 机器学习及应用（开卷final）/final project/文本挖掘/dataset/1.original-aclImdb_v1.tar.gz/aclImdb/"
    file_list=[]

    positive_path=path + filetype+"/pos/"
    for f in os.listdir(positive_path):
        file_list+=[positive_path+f]
    
    negative_path=path + filetype+"/neg/"
    for f in os.listdir(negative_path):
        file_list+=[negative_path+f]
        
    print('read',filetype, 'files:',len(file_list))
       
    all_labels = ([1] * 12500 + [0] * 12500) 
        
    all_texts  = []
    for fi in file_list:
        with open(fi,encoding='utf8') as file_input:
            all_texts += [rm_htmltags(" ".join(file_input.readlines()))]
    return all_labels,all_texts

y_train,train_text=read_files("train") 
y_test,test_text=read_files("test")

'''
3.
Build a dictionary to store the words,
transfer the character texts into numeric(int) format
'''
token = Tokenizer(num_words=2000)
token.fit_on_texts(train_text) #word index，words from train_text as key，1 to 2000 as value
oken.word_index #shows the dictionary

x_train_seq = token.texts_to_sequences(train_text)
x_test_seq = token.texts_to_sequences(test_text)

'''
4.
Change the length of element in list into the same,
in order to satisfy the requirement of keras.
'''
x_train = sequence.pad_sequences(x_train_seq,maxlen=100)
x_test = sequence.pad_sequences(x_test_seq,maxlen=100)



#5.Build a sequtial model 
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation,Flatten
from keras.layers.embeddings import Embedding

#embedding
model = Sequential()
model.add(Embedding(output_dim=32,
                    input_dim=2000, 
                    input_length=100))
model.add(Dropout(0.2))#dropout to avoid overfit

#flatten layer
model.add(Flatten())

#hidden layer
model.add(Dense(units=256,
                activation='relu' ))
model.add(Dropout(0.2))#dropout to avoid overfit

#output layer
model.add(Dense(units=1,
                activation='sigmoid' ))
#summary
model.summary()
model.compile(loss='binary_crossentropy',metrics=['accuracy'],optimizer='adam')

#6.Train the model
train_history=model.fit(x_train, y_train,batch_size=100,
                         epochs=10,verbose=2,validation_split=0.2)
                      
#7.Evaluate model
scores = model.evaluate(x_test, y_test, verbose=1)
scores[1]#test result

#8.Predict
predict=model.predict_classes(x_test)
predict_classes=predict.reshape(-1)

#check single prediction result
ResultDict={1:'positive',0:'negtive'}
def show_test_result(i):
    print(test_text[i])
    print('label:',ResultDict[y_test[i]],'prediction result:',ResultDict[predict_classes[i]])
show_test_result(12000)

