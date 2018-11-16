import re, os
import numpy as np
from tqdm import tqdm
from glove import Corpus, Glove
from multiprocessing import Pool
from scipy import spatial


if __name__ == "__main__":
    processed_tweets = open("tweets.txt", "r").readlines()
    sentences = [['<UKN>']]
    for line in processed_tweets:
        sent = line.strip().split(' ')
        sentences.append(sent)
    corpus = Corpus()
    # corpus.fit(sentences, window = 3)    # window parameter denotes the distance of context
    # glove = Glove(no_components = 100, learning_rate = 0.05)
    # glove.fit(matrix = corpus.matrix, epochs = 30, no_threads = Pool()._processes, verbose = True)
    # glove.add_dictionary(corpus.dictionary)    #  supply a word-id dictionary to allow similarity queries
    # if os.path.exists('../model') == False:
    #     os.mkdir('../model')
    # glove.save('../model/glove_model')