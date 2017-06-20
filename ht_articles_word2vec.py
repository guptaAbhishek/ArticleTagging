import csv
import numpy as np
import logging
import os.path
import sys
import multiprocessing

from gensim.models import Word2Vec
from gensim.models import word2vec
from gensim.models.word2vec import LineSentence
from gensim.models import Phrases
from gensim.models.phrases import Phraser



def main():

    # program = os.path.basename(sys.argv[0])
    # logger = logging.getLogger(program)
    # logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s')
    # logging.root.setLevel(level=logging.INFO)
    # logger.info("running %s" % ' '.join(sys.argv))
    #
    inp = sys.argv[1]
    # outp1 = sys.argv[2]
    #
    # # unigram word2vec model
    # model = Word2Vec(LineSentence(inp),size = 400,window = 5,min_count = 50,
    #                  workers=multiprocessing.cpu_count())
    #
    # model.save(outp1)
    # model.wv.save_word2vec_format(outp1,binary=False)
    #
    num_features = 300  # Word vector dimensionality
    min_word_count = 50  # Minimum word count
    num_workers = 4  # Number of threads to run in parallel
    context = 6  # Context window size
    downsampling = 1e-3  # Downsample setting for frequent words

    # bigram word2vec model



    # documents = ["the mayor of new york was there", "machine learning can be useful sometimes",
    #              "new york mayor was present"]

    f = file('temp.txt')

    # sentence_stream = [doc.split(" ") for doc in inp]
    # print sentence_stream
    # bigram = Phrases(sentence_stream, min_count=1, threshold=2)
    # sent = [u'the', u'mayor', u'of', u'new', u'york', u'was', u'there']
    # print(bigram[sent])

    bigramer = Phrases(LineSentence(inp))
    bigram_wv_model = Word2Vec(bigramer[LineSentence(inp)], workers=num_workers, \
                     size=num_features, min_count=min_word_count, \
                     window=context, sample=downsampling)

    model_name = 'ht.en.bigram.word2vec.model'
    bigram_wv_model.save(model_name)
    bigram_wv_model.wv.save_word2vec_format(model_name)

    # Trigram word2vec model

    # trigramer = Phrases(bigramer[LineSentence(inp)])
    #
    # trigram_wv_model = Word2Vec(trigramer[LineSentence], workers=num_workers, \
    #                  size=num_features, min_count=min_word_count, \
    #                  window=context, sample=downsampling)
    #
    # model_name_n = 'ht.en.trigram.word2vec.model'
    # trigram_wv_model.save(model_name_n)
    # trigram_wv_model.wv.save_word2vec_format(model_name_n)

if __name__ == '__main__':
    main()