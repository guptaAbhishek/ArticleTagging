import csv
import numpy as np
from HTMLParser import HTMLParser
import logging
import os.path
import sys
import string

from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence

class MLStripper(HTMLParser):
    def __init__(self):
        self.reset()
        self.fed = []
    def handle_data(self, d):
        self.fed.append(d)
    def get_data(self):
        return ''.join(self.fed)

def strip_tags(html):
    s = MLStripper()
    s.feed(html)
    return s.get_data()

def main():
    import pandas as pd
    program = os.path.basename(sys.argv[0])
    logger = logging.getLogger(program)
    logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s')
    logging.root.setLevel(level=logging.INFO)
    logger.info("running %s" % ' '.join(sys.argv))

    outputFile = sys.argv[1]
    i =0
    space = ""
    output = open(outputFile,'w')
    tp = pd.read_csv('MongoDB.csv',sep=',',iterator=True,chunksize=10000)
    df = pd.concat(tp,ignore_index=True)

    for text in df['text'].tolist():
        output.write(space.join(strip_tags(text).translate(None,string.punctuation))+'\n')
        i = i + 1
        if (i % 100 == 0):
            logger.info("Saved " + str(i) + " articles")

    output.close()
    logger.info("Finished Saved"+str(i)+" articles")


if __name__ == '__main__':
    main()