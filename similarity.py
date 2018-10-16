#!/usr/bin/python
# -*- coding: utf-8 -*-
from gensim.scripts.glove2word2vec import glove2word2vec
from gensim.models import KeyedVectors
glove_input_file = 'vectors.txt'
word2vec_output_file = 'word2vec.txt'
glove2word2vec(glove_input_file, word2vec_output_file)
# load the Stanford GloVe model
filename = 'word2vec.txt'
model = KeyedVectors.load_word2vec_format(filename, binary=False)
#print('done')
# calculate: (king - man) + woman = ?
result = model.similarity('కోపం'.decode('utf-8'),'సంతోషం'.decode('utf-8'))
print(result)
