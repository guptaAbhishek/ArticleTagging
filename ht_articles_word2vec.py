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

    inp = sys.argv[1]
    outp1 = sys.argv[2]
    #
    # # unigram word2vec model
    # model = Word2Vec(LineSentence(inp),size = 400,window = 5,min_count = 50,
    #                  workers=multiprocessing.cpu_count())
    #
    # model.save(outp1)
    # model.wv.save_word2vec_format(outp1,binary=False)
    #
    # num_features = 300  # Word vector dimensionality
    # min_word_count = 50  # Minimum word count
    # num_workers = 4  # Number of threads to run in parallel
    # context = 6  # Context window size
    # downsampling = 1e-3  # Downsample setting for frequent words

    # bigram word2vec model



    # documents = ["the mayor of new york was there", "machine learning can be useful sometimes",
    #              "new york mayor was present"]
    sent = ['', 'Mumbais', 'metal', 'music', 'scene', 'has', 'got', 'a', 'shot', 'in', 'the', 'arm', 'with', 'new', 'regular', 'gigs', 'at', 'various', 'venues', 'But', 'in', 'the', 'times', 'of', 'EDM', 'can', 'it', 'survive', 'Last', 'month', 'a', 'highonenergy', 'gig', 'took', 'place', 'at', 'Cult', 'Lounge', 'in', 'Belapur', 'Navi', 'Mumbai', 'Citybased', 'bands', 'like', 'Primitiv', 'and', 'Abstruse', 'performed', 'at', 'the', 'third', 'edition', 'of', 'Black', 'Blood', 'Metalheads', 'moshed', 'like', 'there', 'was', 'no', 'tomorrow', 'Some', 'bled', 'some', 'fell', 'down', 'some', 'fell', 'on', 'the', 'stage', 'All', 'in', 'all', 'a', 'successful', 'metal', 'concert', 'Black', 'Blood', '3', 'is', 'one', 'of', 'a', 'handful', 'of', 'new', 'metalonly', 'music', 'properties', 'in', 'Mumbai', 'Hard', 'Rock', 'Caf', 'the', 'Worli', 'outlet', 'now', 'dedicates', 'the', 'last', 'Sunday', 'of', 'each', 'month', 'to', 'metal', 'Social', 'Khar', 'features', 'the', 'genre', 'exclusively', 'in', 'its', 'basement', 'venue', 'Anti', 'Social', '', 'Jay', 'Singh', 'cofounder', 'and', 'executive', 'director', 'JSM', 'Corporation', 'they', 'run', 'Hard', 'Rock', 'Caf', 'reckons', 'theres', 'demand', 'Its', 'a', 'genre', 'with', 'a', 'lot', 'of', 'passionate', 'fans', 'We', 'wanted', 'to', 'give', 'them', 'a', 'place', 'I', 'see', 'it', 'filling', 'a', 'void', 'in', 'our', 'local', 'music', 'space', 'Evidently', 'theyre', 'not', 'the', 'only', 'ones', 'The', 'decline', '', 'In', '2012', 'underground', 'venue', 'B69', 'in', 'Andheri', 'shut', 'its', 'doors', 'Metal', 'fans', 'mourned', 'the', 'loss', 'of', 'their', 'favourite', 'most', 'dedicated', 'space', 'for', 'hard', 'rock', 'Himanshu', 'Vaswani', 'business', 'head', 'Bajaao', 'Entertainment', 'the', 'events', 'arm', 'of', 'online', 'musical', 'instruments', 'retailer', 'Bajaoocom', 'recalls', 'There', 'was', 'literally', 'just', 'a', 'stage', 'and', 'enough', 'place', 'for', 'people', 'to', 'stand', 'in', 'there', 'People', 'could', 'walk', 'in', 'dressed', 'however', 'they', 'wanted', 'B69', 'had', 'hosted', 'over', 'a', 'hundred', 'shows', 'It', 'was', 'a', 'death', 'blow', 'for', 'metal', '', 'For', 'most', 'other', 'genres', 'of', 'music', 'demand', 'leads', 'to', 'supply', 'and', 'venues', 'are', 'quick', 'to', 'adapt', 'their', 'programming', 'to', 'match', 'whatever', 'is', 'the', 'current', 'trend', 'Its', 'hardly', 'surprising', 'then', 'that', 'every', 'other', 'restaurant', 'pub', 'and', 'live', 'music', 'space', 'today', 'has', 'a', 'heavy', 'dose', 'of', 'EDM', 'Electronic', 'Dance', 'Music', 'in', 'their', 'lineup', 'The', 'rise', 'and', 'rise', 'of', 'EDM', 'was', 'a', 'big', 'blow', 'to', 'all', 'other', 'genres', '', 'Also', 'read', 'Rise', 'of', 'the', 'machines', 'is', 'EDM', 'pushing', 'other', 'genres', 'out', '', 'Unlike', 'rock', 'it', 'makes', 'for', 'easy', 'listening', 'it', 'doesnt', 'require', 'actual', 'instruments', 'or', 'lyrics', 'Besides', 'EDM', 'is', 'cheaper', 'to', 'produce', 'and', 'more', 'profitable', 'for', 'organisers', 'And', 'as', 'India', 'emerged', 'as', 'a', 'top', 'global', 'market', 'for', 'EDM', 'festivals', 'city', 'venues', 'lost', 'no', 'time', 'in', 'catering', 'to', 'the', 'masses', 'Its', 'back', 'Really', '', 'Despite', 'stiff', 'competition', 'from', 'the', 'nowmammoth', 'EDM', 'scene', 'metal', 'is', 'slowly', 'but', 'surely', 'resurfacing', 'In', 'May', 'Anti', 'Social', 'and', 'Bajaao', 'Entertainment', 'joined', 'hands', 'to', 'launch', 'Roots', '', 'a', 'gig', 'series', 'that', 'showcases', 'metal', 'bands', 'once', 'a', 'month', 'June', 'saw', 'Mumbais', 'original', 'death', 'metal', 'festival', 'Domination', '', 'The', 'Deathfest', 'first', 'launched', 'in', '2000', 'revived', 'in', '2013', 'return', 'to', 'the', 'city', 'after', 'having', 'been', 'in', 'Hyderabad', 'for', 'a', 'year', 'Also', 'read', 'Musician', 'Sahil', 'Makhija', 'on', 'the', 'Indian', 'metal', 'scene', 'today', '', 'Vaswani', 'of', 'Bajaao', 'reckons', 'the', 'internet', 'gave', 'metal', 'bands', 'the', 'platform', 'theyd', 'lost', 'in', 'the', 'physical', 'world', 'It', 'became', 'easier', 'to', 'find', 'an', 'audience', 'who', 'still', 'liked', 'the', 'same', 'music', 'you', 'do', 'Getting', 'the', 'word', 'out', 'was', 'simpler', 'As', 'the', 'fan', 'base', 'was', 'revived', 'it', 'was', 'time', 'to', 'test', 'waters', 'Last', 'year', 'Bajaao', 'revived', 'B69', 'as', 'BIG69', 'a', 'twoday', 'metal', 'festival', 'at', 'Richardson', '', 'Cruddas', 'Byculla', 'Over', '2000', 'people', 'showed', 'up', 'on', 'each', 'day', 'If', 'proof', 'was', 'needed', 'of', 'whether', 'metal', 'could', 'still', 'sell', 'tickets', 'this', 'was', 'it', '', 'More', 'gigs', 'means', 'more', 'opportunities', 'for', 'new', 'bands', 'to', 'cut', 'their', 'teeth', 'And', 'more', 'variety', 'for', 'music', 'lovers', 'More', 'guns', 'less', 'roses', '', 'But', 'in', 'the', 'case', 'of', 'rock', 'or', 'metal', 'the', 'impact', 'of', 'the', 'music', 'is', 'in', 'live', 'performances', 'unlike', 'pop', 'or', 'EDM', 'that', 'you', 'could', 'still', 'enjoy', 'on', 'headphones', 'That', 'makes', 'the', 'availability', 'of', 'venues', 'critical', 'Plus', 'finances', 'are', 'a', 'problem', 'as', 'the', 'demographic', 'of', 'metal', 'fans', 'is', 'usually', 'college', 'students', 'and', 'professionals', 'in', 'their', 'early', 'twenties', 'Socials', 'culture', 'manager', 'Sumit', 'Vaswani', 'says', 'We', 'get', 'a', 'younger', 'audience', 'at', 'Roots', 'We', 'sold', 'tickets', 'for', 'the', 'first', 'two', 'editions', 'at', 'Rs', '500', 'but', 'then', 'had', 'to', 'slash', 'it', 'to', 'Rs', '300', '', 'Venues', 'seem', 'to', 'have', 'realised', 'that', 'theres', 'an', 'audience', 'Its', 'not', 'mainstream', 'like', 'it', 'was', 'a', 'decade', 'ago', 'But', 'its', 'there', 'and', 'it', 'might', 'grow', 'Irrespective', 'of', 'how', 'things', 'pan', 'out', 'were', 'just', 'glad', 'theres', 'an', 'alternative', 'to', 'the', 'standardissue', 'lyricless', 'EDM', '', 'Upcoming', 'city', 'bands', 'to', 'watch', 'out', 'for', '', '', 'The', 'Minerva', 'Conduct', '', '', 'When', 'you', 'talk', 'of', 'some', 'of', 'the', 'best', 'players', 'in', 'the', 'Mumbai', 'metal', 'scene', 'coming', 'together', 'you', 'think', 'of', 'The', 'Minerva', 'Conduct', 'The', 'core', 'group', 'of', 'the', 'band', 'Ashwin', 'Shriyan', 'Nishith', 'Hegde', 'and', 'Pratheek', 'Rajagopal', 'dish', 'out', 'some', 'crazy', 'experimental', 'progressive', 'metal', 'These', 'guys', 'give', 'you', 'serious', 'envy', 'with', 'their', 'skill', 'levels', 'as', 'musicians', 'and', 'the', 'complexity', 'of', 'their', 'music', '', 'Darkrypt', '', '', 'A', 'promising', 'new', 'band', 'which', 'has', 'just', 'released', 'their', 'debut', 'album', 'Delirious', 'Excursion', 'featuring', 'a', 'host', 'of', 'collaborations', 'with', 'international', 'artists', 'for', 'their', 'artwork', 'lyrics', 'and', 'even', 'the', 'mixing', 'and', 'mastering', 'I', 'saw', 'them', 'live', 'before', 'the', 'album', 'came', 'out', 'and', 'they', 'showed', 'a', 'lot', 'of', 'promise', 'Their', 'album', 'definitely', 'has', 'set', 'them', 'a', 'high', 'benchmark', 'to', 'match', 'up', 'to', 'when', 'they', 'play', 'live', 'now', 'Check', 'it', 'out', 'if', 'death', 'metal', 'is', 'your', 'thing', 'Killchain', '', '', 'Their', 'music', 'is', 'primal', 'and', 'raw', 'death', 'metal', 'It', 'reminds', 'me', 'of', 'Obituary', 'if', 'I', 'have', 'to', 'think', 'of', 'a', 'reference', 'point', 'They', 'are', 'a', 'good', 'live', 'band', 'and', 'are', 'currently', 'working', 'on', 'their', 'EP', 'which', 'Im', 'quite', 'excited', 'to', 'listen', 'to', 'They', 'can', 'definitely', 'be', 'a', 'force', 'to', 'reckon', 'with', 'if', 'they', 'stick', 'around', 'The', 'Calvin', 'Cycle', '', '', 'Honestly', 'they', 'arent', 'the', 'tightest', 'live', 'band', 'but', 'they', 'bring', 'something', 'new', 'to', 'the', 'Mumbai', 'scene', 'with', 'their', 'brand', 'of', 'gothic', 'metal', 'Think', 'Evanescence', 'and', 'Lacuna', 'Coil', 'If', 'these', 'guys', 'can', 'hone', 'their', 'chops', 'they', 'can', 'go', 'a', 'long', 'way', '', '', 'By', 'Sahil', 'Makhija', 'Drummer', 'Reptilian', 'Death', 'and', 'Vocalist', 'and', 'Guitarist', 'Demonic', 'Resurrection', '', '', '\n']

    with open('temp.txt') as f:
        content = f.readline()
        t = [content.split(" ")]
        bigram = Phrases(t, min_count=1, threshold=2)
        print bigram[sent]

    # content = [x.strip() for x in content]

    # print content
        # sentence_stream = [doc.split(" ") for doc in inp]
    # print sentence_stream
    # bigram = Phrases(sentence_stream, min_count=1, threshold=2)
    # sent = [u'the', u'mayor', u'of', u'new', u'york', u'was', u'there']
    # print(bigram[sent])


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