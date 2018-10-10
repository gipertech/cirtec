# -*- coding: utf-8 -*-

""""""

import numpy as np
import pymorphy2
import gensim
import pandas as pd

from scipy.spatial.distance import cosine
from base_defs import get_average_len_words

with open('../initial_data/Word2Vec__fixes.stem.txt', 'rb') as f:
    data = f.read()
text = data.decode('utf-8') # a `str`; this step can't be used if data is binary
del data
print("len_text: {}".format(len(text)))
sentences = [sentence for sentence in text.split("\n") if len(sentence) > 60]
# # debug
# sentences = sentences[:1000]
del text
print("len_sentences: {}".format(len(sentences)))
print(sentences[0])

min_average_len_words = 8
sentences = [[word for word in sentence.split()] for sentence in sentences
             if get_average_len_words(sentence) >= min_average_len_words]

morph = pymorphy2.MorphAnalyzer()
#
# http://rusvectores.org/ru/models/
#     web_upos_cbow_300_20_2017
#
w2v_model = gensim.models.KeyedVectors.load_word2vec_format('../models/web_0_300_20.bin', binary=True)
w2v_model.init_sims(replace=True)

cotags = {
    'ADJF' : 'DET',
    'ADJS' : 'ADJ', 
    'ADVB' : 'ADV', 
    'COMP' : 'ADV', 
    'GRND' : 'VERB', 
    'INFN' : 'VERB', 
    'NOUN' : 'NOUN', 
    'PRED' : 'ADV', 
    'NPRO' : 'PRON', 
    'PREP' : 'ADV', 
    'CONJ' : 'СCONJ',
    'PRTF' : 'ADJ', 
    'PRTS' : 'VERB', 
    'VERB' : 'VERB',
    'INTJ' : 'INTJ',
    'PRCL' : 'PART',
    'NUMR' : 'NUM',
    'None' : 'NOUN',
}

def get_word_with_correct_tag(word_and_tag):
    sp = word_and_tag.split("_")
    if len(sp) != 2:
        print(sp)
    return "_".join((sp[0], cotags[str(sp[1])]))

vectors = [dict(vector=np.mean([w2v_model.get_vector(word) for word in words if word in w2v_model.vocab], axis=0), number=number) for number, words in enumerate([[get_word_with_correct_tag(word) for word in sentence.split()] for sentence in sentences])]

clusters = []

df_vectors = pd.DataFrame(vectors)

threshold = 0.15
min_size_cluster = 100

counter = 0

while df_vectors.shape[0] > 10000:
    vector_0 = df_vectors.vector.iloc[0]
    df_vectors['res'] = df_vectors.vector.apply(lambda vector: cosine(vector_0, vector))
    cluster = df_vectors[df_vectors['res'] <= threshold]
    df_vectors = df_vectors[df_vectors['res'] > threshold]
    if cluster.shape[0] > min_size_cluster:
        clusters.append(cluster)
        counter += 1
        print('shape of new cluster: {:4}, counter: {}, dataset:{}'.format(cluster.shape[0], counter, df_vectors.shape[0]))
        with open("../resulting_data/clustering_analysis_stem__web_0_300_20.bin/{}.csv".format(counter), "w") as f:
            f.writelines(("{}\n".format(sen) for sen in (" ".join([word.split("_")[0] for word in sentences[number].split()]) for number in cluster.number)))
        break
    else:
        # print("drop: {}".format(cluster.shape[0]))
        pass
