import sys
import glob
import errno
import string
import io
from gensim import corpora, models, similarities

def extract_candidate_chunks(text,grammer=r'KT: {(<JJ>* <NN.*>+ <IN>)? <JJ>* <NN.*>+}'):
    import itertools, nltk, string

    punct = set(string.punctuation)
    stop_words = set(nltk.corpus.stopwords.words('english'))

    chunker = nltk.chunk.regexp.RegexpParser(grammer)

    tagged_sents = nltk.pos_tag_sents(nltk.word_tokenize(sent) for sent in nltk.sent_tokenize(text))

    all_chunks = list(itertools.chain.from_iterable(nltk.chunk.tree2conlltags(chunker.parse(tagged_sent))
                                                    for tagged_sent in tagged_sents))


    candidates = [' '.join(word for word,pos,chunk in group).lower()
                  for key, group in itertools.groupby(all_chunks, lambda (word,pos,chunk): chunk != '0') if key]

    return [cand for cand in candidates
            if cand not in stop_words and not all(char in punct for char in cand)]



def extract_candidate_words(text,good_tags=set(['JJ','JJR','JJS','NN','NNP','NNS','NNPS'])):
    import itertools,nltk,string

    punct = set(string.punctuation)
    stop_words = set(nltk.corpus.stopwords.words('english'))

    tagged_words = itertools.chain.from_iterable(nltk.pos_tag_sents(nltk.word_tokenize(sent)
                                                                    for sent in nltk.sent_tokenize(text)))

    candidates = [word.lower() for word, tag in tagged_words
                  if tag in good_tags and word.lower() not in stop_words
                  and not all(char in punct for char in word)]

    return candidates


def score_keyphrases_by_tfidf(texts, candidates='chunks'):
    import gensim, nltk

    # extract candidates from each text in texts, either chunks or words
    if candidates == 'chunks':
        boc_texts = [extract_candidate_chunks(text) for text in texts]
    elif candidates == 'words':
        boc_texts = [extract_candidate_words(text) for text in texts]
    # make gensim dictionary and corpus
    dictionary = gensim.corpora.Dictionary(boc_texts)
    corpus = [dictionary.doc2bow(boc_text) for boc_text in boc_texts]
    # transform corpus with tf*idf model
    tfidf = gensim.models.TfidfModel(corpus)
    corpus_tfidf = tfidf[corpus]


    return corpus_tfidf ,dictionary





def main():
    texts = []
    path = 'docs/*.txt'
    files = glob.glob(path)

    for name in files:
        try:
            with open(name) as f:
                content = f.readline()
                content = ''.join([c for c in content if c not in ("!", "?","'",".",",",":","`")])
                texts.append(content)
        except IOError as exc:  # Not sure what error this is
            if exc.errno != errno.EISDIR:
                raise


    # print extract_candidate_chunks(texts[0])
    print score_keyphrases_by_tfidf(texts)


if __name__ == '__main__':
    main()