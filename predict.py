import pickle
import numpy as np
from tqdm import tqdm
from glove import Corpus, Glove
from sklearn.neighbors import KNeighborsClassifier 
from random import shuffle


# define a function that converts word into embedded vector
def vector_converter(glove, word):
    if word not in glove.dictionary:
        idx = glove.dictionary['<UKN>']
    else:
        idx = glove.dictionary[word]
    return glove.word_vectors[idx]

def embedding(tweets, glove):
    tweetEmbeddings = np.zeros((1,100))
    for tweet in tqdm(tweets):
        words = tweet.split(' ')
        vec = np.zeros((1, 100))
        for word in words:
            vec += vector_converter(glove, word.strip())
        vec /= len(words)
        tweetEmbeddings = np.append(tweetEmbeddings, vec, axis = 0)
    return tweetEmbeddings[1:]

def processY(labels):
    Y = np.zeros((len(labels), max(labels)))
    i = 0
    for label in labels:
        Y[i][label] = 1
    return Y

if __name__ == "__main__":
    glove = Glove.load('../model/glove_model')

    tweets = open("tweets.txt", "r").readlines()
    shuffle(tweets)

	tweets_train = tweets[0:(int)(0.8 * len(tweets))]
	tweets_test = tweets[(int)(0.8 * len(tweets)):]

	X_train = []
	Y_train = []
	for tweet in tweets_train:
		tweet = tweet.split("%^&")
		X_train.append(tweet[0])
		Y_train.append(tweet[1])

	print('Training Data Loading...')
	X_train = embedding(X_train, glove)
    Y_train = np.array(Y_train)

	X_test = []	
	Y_test = []
	for tweet in tweets_test:
		tweet = tweet.split("%^&")
		X_test.append(tweet[0])
		Y_test.append(tweet[1])

    print('Testing Data Loading...')
	X_test = embedding(X_test, glove)
    Y_test = np.array(Y_test)

    pklFile = open('../model/X_train.pkl', 'ab')
    pickle.dump(X_train, pklFile)
    pklFile.close()

    pklFile = open('../model/Y_train.pkl', 'ab')
    pickle.dump(Y_train, pklFile)
    pklFile.close()

    pklFile = open('../model/X_test.pkl', 'ab')
    pickle.dump(X_test, pklFile)
    pklFile.close()

    pklFile = open('../model/Y_test.pkl', 'ab')
    pickle.dump(Y_test, pklFile)
    pklFile.close()

    knn = KNeighborsClassifier(n_neighbors = 7).fit(X_train, Y_train)
    accuracy = knn.score(X_test, Y_test) 
    print("Accuracy is ", accuracy)