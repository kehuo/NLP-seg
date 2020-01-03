# -*- coding: utf-8 -*-
import codecs
import gensim

def get_sentences(rfile):
    sentences = []
    f = codecs.open(rfile, encoding='utf-8', mode='r')
    for line in f:
        line.rstrip()
        sentences.append(line)
    f.close()
    return sentences

sentences = get_sentences("/home/jerry.ji/nlp_data/emr.txt")
print("read %d sentences" % (len(sentences)))
model = gensim.models.Word2Vec(sentences, size=100, window=5, min_count=0, workers=4)
model.wv.save_word2vec_format("/home/jerry.ji/nlp_data/vectors.txt")
