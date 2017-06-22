import csv
import numpy as np
from HTMLParser import HTMLParser
import logging
import os.path
import sys
import string
import unicodedata
import sys



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

    return


def readMongoCSV():

    return

def main():
    readMongoCSV()
    return


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

    bigram = Phrases()
    z = 0
    testing_sent = ""
    for article in collection.find({},{"textStrip":1,"_id":0}):
        z = z+1
        if z == 1000:
            print 'now '+str(z)
            lemma_words = []
            if article['textStrip'] != "":
                if len(article['textStrip'].split()) >5:
                    textStripWithHTML = article['textStrip']
                    textStringWithoutHTML = strip_tags(textStripWithHTML)
                    print textStringWithoutHTML
                    s = "".join(l for l in textStringWithoutHTML if l not in string.punctuation)
                    s = "".join([i for i in s if not i.isdigit()])
                    s = " ".join([i for i in s.lower().split() if i not in stop])
                    s = nltk.word_tokenize(s)
                    sent_pos_tags = nltk.pos_tag(s)

                    for i in sent_pos_tags:
                        if str(penn_to_wn(i[1])) != "None":
                            lemma_words.append(nltk.stem.WordNetLemmatizer().lemmatize(i[0], pos=penn_to_wn(i[1])))
                            print lemma_words
                            return
                            bigram.add_vocab(lemma_words)
                    # output.write(getSent(lemma_words)+'\n')
                    testing_sent = lemma_words

            z = z +1
            c = c + 1
            if (c % 100 == 0):
                logger.info("Saved " + str(c) + " articles")
            break
    # sent = [u'clone', u'hoax', u'scientific', u'method', u'raelien', u'sect', u'claim', u'really', u'manage', u'produce', u'baby', u'clone', u'name', u'eve', u'nothing', u'simple', u'dna', u'test', u'dna', u'sample', u'determine', u'eve', u'clone', u'mother', u'take', u'tuesday', u'babys', u'return', u'parent', u'home', u'unspecified', u'location', u'say', u'brigitte', u'boisselier', u'president', u'clonaid', u'human', u'clone', u'company', u'found', u'raeliens', u'boisselier', u'friday', u'make', u'startle', u'announcement', u'eve', u'bear', u'december', u'claim', u'true', u'eve', u'carbon', u'copy', u'mother', u'age', u'difference', u'year', u'boisselier', u'authorize', u'abc', u'television', u'science', u'editor', u'michael', u'guillen', u'team', u'expert', u'test', u'eve', u'really', u'clone', u'skin', u'cell', u'mother', u'test', u'standard', u'paternity', u'lawsuit', u'police', u'investigation', u'identify', u'suspect', u'sample', u'dna', u'take', u'crime', u'scenes', u'eve', u'case', u'procedure', u'simple', u'blood', u'sample', u'take', u'mucus', u'membrane', u'mouth', u'mother', u'child', u'enough', u'make', u'genetic', u'profile', u'nuclear', u'dna', u'somatic', u'cell', u'donor', u'baby', u'match', u'order', u'absolutely', u'certain', u'cloning', u'perform', u'say', u'william', u'muir', u'genetics', u'professor', u'purdue', u'university', u'west', u'lafayette', u'indiana', u'discover', u'dna', u'deoxyribonucleic', u'acid', u'form', u'basic', u'material', u'chromosome', u'cell', u'nucleus', u'contain', u'genetic', u'code', u'transmit', u'person', u'hereditary', u'pattern', u'genetic', u'code', u'dictate', u'production', u'protein', u'virtually', u'identical', u'everyone', u'however', u'cent', u'base', u'pair', u'call', u'intron', u'apparently', u'arrange', u'random', u'make', u'individual', u'unique', u'sequence', u'identification', u'perform', u'different', u'technology', u'common', u'rflp', u'test', u'use', u'identify', u'paternity', u'link', u'pcr', u'test', u'mainly', u'use', u'criminal', u'case', u'restriction', u'fragment', u'length', u'polymorphism', u'rflp', u'test', u'use', u'internationally', u'large', u'dna', u'sample', u'require', u'test', u'accuracy', u'high', u'cent', u'commonly', u'use', u'technology', u'polymerase', u'chain', u'reaction', u'pcr', u'test', u'marginally', u'less', u'accurate', u'rflp', u'test', u'pcr', u'test', u'still', u'reliable', u'prove', u'useful', u'criminology', u'require', u'tiny', u'quantity', u'dna', u'even', u'poor', u'condition', u'millilitre', u'blood', u'need', u'test', u'require', u'minimum', u'age', u'carry', u'unborn', u'child', u'test', u'perform', u'use', u'amniotic', u'liquid', u'instead', u'blood', u'dna', u'test', u'cost', u'less', u'depend', u'laboratory', u'protocol', u'follow', u'dr', u'guillen', u'available', u'say', u'dna', u'test', u'method', u'use', u'identify', u'expert', u'administer']
    print testing_sent
    print(bigram[testing_sent])

    output.close()
    logger.info("Finished Saved"+str(c)+" articles")


if __name__ == '__main__':
    main()