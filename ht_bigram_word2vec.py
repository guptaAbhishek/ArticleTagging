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

    program = os.path.basename(sys.argv[0])
    logger = logging.getLogger(program)
    logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s')
    logging.root.setLevel(level=logging.INFO)
    logger.info("running %s" % ' '.join(sys.argv))








    # inp = sys.argv[1]
    # outp1 = sys.argv[2]

    num_features = 300  # Word vector dimensionality
    min_word_count = 50  # Minimum word count
    num_workers = 4  # Number of threads to run in parallel
    context = 6  # Context window size
    downsampling = 1e-3  # Downsample setting for frequent words


    # bigram = Phrases(LineSentence(inp), min_count=1, threshold=2)

    #
    #
    # bigram_wv_model = Word2Vec(bigram[LineSentence(inp)], workers=num_workers, \
    #                  size=num_features, min_count=min_word_count, \
    #                  window=context, sample=downsampling)
    #
    # model_name = 'ht.en.bigram.word2vec.model'
    # bigram_wv_model.save(model_name)
    # bigram_wv_model.wv.save_word2vec_format(model_name)

if __name__ == '__main__':
    main()