# -*- coding: utf-8 -*-

from gensim.scripts.glove2word2vec import glove2word2vec
#from gensim.models import Word2Vec
from gensim.models import KeyedVectors



'''
f=open('evaluation.txt','r')
f2 = open('eval.txt','w')

for line in f:
    for ch in line:
        if ch not in [' ']:
            f2.write(ch)

f.close()
f2.close()

'''
import pandas
import numpy as np

a = pandas.read_csv('eval.txt', sep= ',',header= 0)
#model = KeyedVectors.load_word2vec_format('final_trial_1', binary=False, unicode_errors= 'ignore')
glove_input_file = 'vectors.txt'
word2vec_output_file = 'word2vec.txt'
glove2word2vec(glove_input_file, word2vec_output_file)
filename = 'word2vec.txt'
model = KeyedVectors.load_word2vec_format(filename, binary=False)
#print('done')
# calculate: (king - man) + woman = ?
#result = model.similarity('కోపం'.decode('utf-8'),'సంతోషం'.decode('utf-8'))
#print(result)

pred =[]
j=0
sq_error = 0

for i in a.index:
    pred.append(model.similarity(a['word_1'].iloc[i].decode('utf-8'),a['word_2'].iloc[i].decode('utf-8')))

pred = np.array(pred)

#tr =  pred>= 0.5

#pred = [1 if i==True else 0 for i in tr]


loss = 0
for i in range(len(pred)):
    loss  = loss + np.absolute(a['similarity'].iloc[i] - pred[i])

print('loss is: ', float(loss)/len(pred))



