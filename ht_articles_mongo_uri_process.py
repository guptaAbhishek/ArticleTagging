import csv
import numpy as np
from HTMLParser import HTMLParser
import logging
import os.path
import sys
import string
import unicodedata



from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence
from gensim.models import Phrases
from gensim.models.phrases import Phraser



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

from nltk.corpus import wordnet as wn

def is_noun(tag):
    return tag in ['NN', 'NNS', 'NNP', 'NNPS']


def is_verb(tag):
    return tag in ['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ']


def is_adverb(tag):
    return tag in ['RB', 'RBR', 'RBS']


def is_adjective(tag):
    return tag in ['JJ', 'JJR', 'JJS']


def penn_to_wn(tag):
    if is_adjective(tag):
        return wn.ADJ
    elif is_noun(tag):
        return wn.NOUN
    elif is_adverb(tag):
        return wn.ADV
    elif is_verb(tag):
        return wn.VERB
    return None

def extract_ner(text):
    from nltk import ne_chunk, pos_tag, word_tokenize
    from nltk.tree import Tree

    chunked = ne_chunk(pos_tag(word_tokenize(text)))
    prev = None
    continuous_chunk = []
    current_chunk = []

    for i in chunked:
        if type(i) == Tree:
            current_chunk.append(" ".join([token for token, pos in i.leaves()]))
        elif current_chunk:
            named_entity = " ".join(current_chunk)
            if named_entity not in continuous_chunk:
                continuous_chunk.append(named_entity)
                current_chunk = []
            else:
                continue
    return continuous_chunk


def getSent(lemma_words):
    space = " "
    r =""
    for w in lemma_words:
        r = space.join(i for i in lemma_words)

    return r

def main():
    import nltk
    from pymongo import MongoClient
    from nltk.corpus import stopwords
    from string import maketrans
    from nltk.stem.lancaster import LancasterStemmer
    from nltk.stem import WordNetLemmatizer


    program = os.path.basename(sys.argv[0])
    logger = logging.getLogger(program)
    logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s')
    logging.root.setLevel(level=logging.INFO)
    logger.info("running %s" % ' '.join(sys.argv))
    space = ""

    stop = set(stopwords.words('english'))
    client = MongoClient('mongodb://10.1.2.83:27017')
    db = client.db_htfeedengine
    collection = db.article
    real_tags = []
    list_of_sent = []

    outputFile = sys.argv[1]
    i = 0
    space = ""
    c = 0
    output = open(outputFile, 'w')

    sentences = []
    bigram = Phrases()

    for article in collection.find({},{"textStrip":1,"_id":0}):
        lemma_words = []
        if article['textStrip'] != "":
            if len(article['textStrip'].split()) >5:
                textStripWithHTML = article['textStrip']
                textStringWithoutHTML = strip_tags(textStripWithHTML)
                s = "".join(l for l in textStringWithoutHTML if l not in string.punctuation)
                s = "".join([i for i in s if not i.isdigit()])
                s = " ".join([i for i in s.lower().split() if i not in stop])
                s = nltk.word_tokenize(s)
                sent_pos_tags = nltk.pos_tag(s)


                for i in sent_pos_tags:
                    if str(penn_to_wn(i[1])) != "None":
                        lemma_words.append(nltk.stem.WordNetLemmatizer().lemmatize(i[0], pos=penn_to_wn(i[1])))

                output.write(getSent(lemma_words)+'\n')

                    # print LineSentence(getSent(lemma_words))


                c = c + 1
                if (c % 1000 == 0):
                    logger.info("Saved " + str(c) + " articles")

    output.close()
    logger.info("Finished Saved"+str(c)+" articles")


if __name__ == '__main__':
    main()