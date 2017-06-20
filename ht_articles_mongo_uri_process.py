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



def getDB():
    mongoDbUri = 'mongodb://10.1.2.83:27017'
    from pymongo import MongoClient
    client = MongoClient(mongoDbUri)

    db = client.db_htfeedengine
    return db

def getArticles(db):
    return db.article

def getRows(collections):
    return collections.getCollection()

def main():
    import pandas as pd
    program = os.path.basename(sys.argv[0])
    logger = logging.getLogger(program)
    logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s')
    logging.root.setLevel(level=logging.INFO)
    logger.info("running %s" % ' '.join(sys.argv))

    # outputFile = sys.argv[1]
    # i =0
    # space = ""
    # output = open(outputFile,'w')

    # reading mongodb uri
    # db = getDB()
    # collections = getArticles(db)
    from pymongo import MongoClient
    client = MongoClient('mongodb://10.1.2.83:27017')
    db = client.db_htfeedengine
    collection = db.article
    for article in collection.find({},{"textStrip":1,"_id":0}):
        print strip_tags(article['textStrip'])


    # output.close()
    # logger.info("Finished Saved"+str(i)+" articles")


if __name__ == '__main__':
    main()