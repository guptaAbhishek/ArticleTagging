from sklearn.feature_extraction.text import TfidfVectorizer
import sys
import logging
import os.path
import string
import csv
import numpy as np
from HTMLParser import HTMLParser

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


def getSent(lemma_words):
    space = " "
    r =""
    for w in lemma_words:
        r = space.join(i for i in lemma_words)

    return r



def main():
    reload(sys)
    sys.setdefaultencoding('utf-8')


    tokenize = lambda doc: doc.lower().split(" ")
    import pandas as pd
    from nltk.corpus import stopwords

    import nltk
    from nltk.corpus import stopwords
    from string import maketrans
    from nltk.stem.lancaster import LancasterStemmer
    from nltk.stem import WordNetLemmatizer

    program = os.path.basename(sys.argv[0])
    logger = logging.getLogger(program)
    logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s')
    logging.root.setLevel(level=logging.INFO)
    logger.info("running %s" % ' '.join(sys.argv))

    stop = set(stopwords.words('english'))
    outputFile = sys.argv[1]
    i = 0
    space = ""

    output = open(outputFile, 'w')
    tp = pd.read_csv('MongoDB.csv', sep=',', iterator=True, chunksize=10000,encoding='utf-8')
    df = pd.concat(tp, ignore_index=True)
    all_documents = []
    document = ""

    c = 0
    for text in df['text'].tolist():
        lemma_words= []
        if text != "":
            if len(text.split()) > 5:
                textStripWithHTML = text
                textStringWithoutHTML = strip_tags(textStripWithHTML)
                s = "".join(l for l in textStringWithoutHTML if l not in string.punctuation)
                s = "".join([i for i in s if not i.isdigit()])
                s = " ".join([i for i in s.lower().split() if i not in stop])
                s = nltk.word_tokenize(s)
                sent_pos_tags = nltk.pos_tag(s)

                for i in sent_pos_tags:
                    if str(penn_to_wn(i[1])) != "None":
                        # print nltk.stem.WordNetLemmatizer().lemmatize(i[0], pos=penn_to_wn(i[1]))
                        lemma_words.append(nltk.stem.WordNetLemmatizer().lemmatize(i[0], pos=penn_to_wn(i[1])))

                output.write(getSent(lemma_words)+'\n')
                # output.write(s+'\n')

                # document = s
                # all_documents.append(document)
                c = c + 1
                if (c % 100 == 0):
                    logger.info("Saved " + str(c) + " articles")

    output.close()
    logger.info("Finished Saved" + str(i) + " articles")

    # sklearn_tfidf = TfidfVectorizer(norm='l2',min_df=0, use_idf=True, smooth_idf=False, sublinear_tf=True, tokenizer=tokenize)

    # sklearn_representation = sklearn_tfidf.fit_transform(all_documents)


    # print sklearn_representation[0]
    # print sklearn_representation.toarray()[0].tolist()



if __name__ == '__main__':
    main()