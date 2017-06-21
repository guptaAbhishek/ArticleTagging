import sys
import glob
import errno
import string
import io
from gensim import corpora, models, similarities
from string import digits

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


def score_keyphrases_by_tfidf(texts, candidates='words'):
    import gensim, nltk

    # extract candidates from each text in texts, either chunks or words
    if candidates == 'chunks':
        boc_texts = [extract_candidate_chunks(text) for text in texts]
    elif candidates == 'words':
        boc_texts = [extract_candidate_words(text) for text in texts]


    # make gensim dictionary and corpus
    # print texts[0]
    # print extract_candidate_words(texts[0])
    # print boc_texts
    dictionary = gensim.corpora.Dictionary(boc_texts)
    corpus = [dictionary.doc2bow(boc_text) for boc_text in boc_texts]
    # print corpus
    # transform corpus with tf*idf model
    tfidf = gensim.models.TfidfModel(corpus)
    corpus_tfidf = tfidf[corpus]


    return corpus_tfidf ,dictionary


def score_keyphrases_by_text_rank(text,n_keywords=0.05):
    from itertools import takewhile, tee, izip
    import networkx, nltk

    # tokenize for all words, and extract *candidate* words
    words = [word.lower()
             for sent in nltk.sent_tokenize(text)
             for word in nltk.word_tokenize(sent)]
    candidates = extract_candidate_words(text)
    # build graph, each node is a unique candidate
    graph = networkx.Graph()
    graph.add_nodes_from(set(candidates))

    # iterate over word-pairs, add unweighted edges into graph
    def pairwise(iterable):
        """s -> (s0,s1), (s1,s2), (s2, s3), ..."""
        a, b = tee(iterable)
        next(b, None)
        return izip(a, b)

    for w1, w2 in pairwise(candidates):
        if w2:
            graph.add_edge(*sorted([w1, w2]))
    # score nodes using default pagerank algorithm, sort by score, keep top n_keywords
    ranks = networkx.pagerank(graph)
    if 0 < n_keywords < 1:
        n_keywords = int(round(len(candidates) * n_keywords))
    word_ranks = {word_rank[0]: word_rank[1]
                  for word_rank in sorted(ranks.iteritems(), key=lambda x: x[1], reverse=True)[:n_keywords]}
    keywords = set(word_ranks.keys())
    # merge keywords into keyphrases
    keyphrases = {}
    j = 0
    for i, word in enumerate(words):
        if i < j:
            continue
        if word in keywords:
            kp_words = list(takewhile(lambda x: x in keywords, words[i:i + 10]))
            avg_pagerank = sum(word_ranks[w] for w in kp_words) / float(len(kp_words))
            keyphrases[' '.join(kp_words)] = avg_pagerank
            # counter as hackish way to ensure merged keyphrases are non-overlapping
            j = i + len(kp_words)

    return sorted(keyphrases.iteritems(), key=lambda x: x[1], reverse=True)


def extract_candidate_features(candidates, doc_text, doc_excerpt, doc_title):
    import collections, math, nltk, re


    candidate_scores = collections.OrderedDict()


    # get word counts for document


    doc_word_counts = collections.Counter(word.lower()
                                          for sent in nltk.sent_tokenize(doc_text)
                                          for word in nltk.word_tokenize(sent))


    for candidate in candidates:

        # print candidate
        pattern = re.compile(r'\b' + re.escape(candidate) + r'(\b|[,;.!?]|\s)', re.IGNORECASE)
        # frequency-based
        # number of times candidate appears in document
        cand_doc_count = len(pattern.findall(doc_text))
        # count could be 0 for multiple reasons; shit happens in a simplified example
        if not cand_doc_count:
            print '**WARNING:', candidate, 'not found!'
            continue

        # statistical
        candidate_words = candidate.split()
        max_word_length = max(len(w) for w in candidate_words)
        term_length = len(candidate_words)
        # get frequencies for term and constituent words
        sum_doc_word_counts = float(sum(doc_word_counts[w] for w in candidate_words))
        try:
            # lexical cohesion doesn't make sense for 1-word terms
            if term_length == 1:
                lexical_cohesion = 0.0
            else:
                lexical_cohesion = term_length * (
                1 + math.log(cand_doc_count, 10)) * cand_doc_count / sum_doc_word_counts
        except (ValueError, ZeroDivisionError) as e:
            lexical_cohesion = 0.0

        # positional
        # found in title, key excerpt
        in_title = 1 if pattern.search(doc_title) else 0
        in_excerpt = 1 if pattern.search(doc_excerpt) else 0
        # first/last position, difference between them (spread)
        doc_text_length = float(len(doc_text))
        first_match = pattern.search(doc_text)
        abs_first_occurrence = first_match.start() / doc_text_length
        if cand_doc_count == 1:
            spread = 0.0
            abs_last_occurrence = abs_first_occurrence
        else:
            for last_match in pattern.finditer(doc_text):
                pass
            abs_last_occurrence = last_match.start() / doc_text_length
            spread = abs_last_occurrence - abs_first_occurrence

        candidate_scores[candidate] = {'term_count': cand_doc_count,
                                       'term_length': term_length, 'max_word_length': max_word_length,
                                       'spread': spread, 'lexical_cohesion': lexical_cohesion,
                                       'in_excerpt': in_excerpt, 'in_title': in_title,
                                       'abs_first_occurrence': abs_first_occurrence,
                                       'abs_last_occurrence': abs_last_occurrence}

    return candidate_scores

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

    # score_keyphrases_by_text_rank(texts[0])
    # print texts
    # print extract_candidate_words(texts[0])
    # print score_keyphrases_by_tfidf(texts)
    print texts[0]
    t = ' Mumbais metal music scene has got a shot in the arm with new regular gigs at various venues But in the times of EDM can it survive Last month a highonenergy gig took place at Cult Lounge in Belapur Navi Mumbai Citybased bands like Primitiv and Abstruse performed at the third edition of Black Blood Metalheads moshed like there was no tomorrow Some bled some fell down some fell on the stage All in all a successful metal concert Black Blood 3 is one of a handful of new metalonly music properties in Mumbai Hard Rock Caf the Worli outlet now dedicates the last Sunday of each month to metal Social Khar features the genre exclusively in its basement venue Anti Social  Jay Singh cofounder and executive director JSM Corporation they run Hard Rock Caf reckons theres demand Its a genre with a lot of passionate fans We wanted to give them a place I see it filling a void in our local music space Evidently theyre not the only ones The decline  In 2012 underground venue B69 in Andheri shut its doors Metal fans mourned the loss of their favourite most dedicated space for hard rock Himanshu Vaswani business head Bajaao Entertainment the events arm of online musical instruments retailer Bajaoocom recalls There was literally just a stage and enough place for people to stand in there People could walk in dressed however they wanted B69 had hosted over a hundred shows It was a death blow for metal  For most other genres of music demand leads to supply and venues are quick to adapt their programming to match whatever is the current trend Its hardly surprising then that every other restaurant pub and live music space today has a heavy dose of EDM Electronic Dance Music in their lineup The rise and rise of EDM was a big blow to all other genres  Also read Rise of the machines is EDM pushing other genres out  Unlike rock it makes for easy listening it doesnt require actual instruments or lyrics Besides EDM is cheaper to produce and more profitable for organisers And as India emerged as a top global market for EDM festivals city venues lost no time in catering to the masses Its back Really  Despite stiff competition from the nowmammoth EDM scene metal is slowly but surely resurfacing In May Anti Social and Bajaao Entertainment joined hands to launch Roots  a gig series that showcases metal bands once a month June saw Mumbais original death metal festival Domination  The Deathfest first launched in 2000 revived in 2013 return to the city after having been in Hyderabad for a year Also read Musician Sahil Makhija on the Indian metal scene today  Vaswani of Bajaao reckons the internet gave metal bands the platform theyd lost in the physical world It became easier to find an audience who still liked the same music you do Getting the word out was simpler As the fan base was revived it was time to test waters Last year Bajaao revived B69 as BIG69 a twoday metal festival at Richardson  Cruddas Byculla Over 2000 people showed up on each day If proof was needed of whether metal could still sell tickets this was it  More gigs means more opportunities for new bands to cut their teeth And more variety for music lovers More guns less roses  But in the case of rock or metal the impact of the music is in live performances unlike pop or EDM that you could still enjoy on headphones That makes the availability of venues critical Plus finances are a problem as the demographic of metal fans is usually college students and professionals in their early twenties Socials culture manager Sumit Vaswani says We get a younger audience at Roots We sold tickets for the first two editions at Rs 500 but then had to slash it to Rs 300  Venues seem to have realised that theres an audience Its not mainstream like it was a decade ago But its there and it might grow Irrespective of how things pan out were just glad theres an alternative to the standardissue lyricless EDM  Upcoming city bands to watch out for   The Minerva Conduct   When you talk of some of the best players in the Mumbai metal scene coming together you think of The Minerva Conduct The core group of the band Ashwin Shriyan Nishith Hegde and Pratheek Rajagopal dish out some crazy experimental progressive metal These guys give you serious envy with their skill levels as musicians and the complexity of their music  Darkrypt   A promising new band which has just released their debut album Delirious Excursion featuring a host of collaborations with international artists for their artwork lyrics and even the mixing and mastering I saw them live before the album came out and they showed a lot of promise Their album definitely has set them a high benchmark to match up to when they play live now Check it out if death metal is your thing Killchain   Their music is primal and raw death metal It reminds me of Obituary if I have to think of a reference point They are a good live band and are currently working on their EP which Im quite excited to listen to They can definitely be a force to reckon with if they stick around The Calvin Cycle   Honestly they arent the tightest live band but they bring something new to the Mumbai scene with their brand of gothic metal Think Evanescence and Lacuna Coil If these guys can hone their chops they can go a long way   By Sahil Makhija Drummer Reptilian Death and Vocalist and Guitarist Demonic Resurrection '
    print '###'
    print t
    cadidates = score_keyphrases_by_text_rank(t)

    cands = []
    for c in cadidates:
        cands.append(c[0])


    doc_title = "Mumbais metal music scene is getting a revival with new, regular gigs"

    doc_text = t.translate(None, digits)
    doc_excerpt = t.translate(None, digits)
    doc_text = doc_text.translate(None,string.punctuation)
    doc_excerpt = doc_excerpt.translate(None, string.punctuation)

    # extract_candidate_features(cadidates, doc_text, doc_excerpt, doc_title)
    # print extract_candidate_features(cands,doc_text,doc_excerpt,doc_title)
    score_keyphrases_by_tfidf()


if __name__ == '__main__':
    main()