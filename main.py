# TODO load tweets
# TODO load positive and negative tweets as training set
# TODO calculate X-vector of positive/negative values
# TODO label Y - vector of positive/negative labels
# TODO load testing set
# TODO calculate X-vector of positive/negative values for each tweet
# TODO label Y - using knn
# TODO label Y using svm

import nltk
import operator
#function that's returns a dictionary of words as keys and their positivity/negativity as values
def load_words_weight(filename):
    f = open(filename, 'r')
    words_weights = {}
    line = f.readline()
    nbr = 0
    while line:
        nbr += 1
        #        print "%d lines loaderd from afinn" % (nbr)
        l = line[:-1].split('\t')
        words_weights[l[0]] = float(l[1]) / 4  # Normalizing
        line = f.readline()

    return words_weights
#function that's returns a dictionary of emoticons as keys and their positivity/negativity as values
def createEmoticonDictionary(filename):
    emo_scores = {'Positive': 0.5, 'Extremely-Positive': 1.0, 'Negative':-0.5,'Extremely-Negative': -1.0,'Neutral': 0.0}
    emo_score_list={}
    fi = open(filename,encoding="utf8")
    l=fi.readline()
    while l:
        l=l.replace("\xc2\xa0"," ")
        li=l.split(" ")
        l2=li[:-1]
        l2.append(li[len(li)-1].split("\t")[0])
        sentiment=li[len(li)-1].split("\t")[1][:-1]
        score=emo_scores[sentiment]
        l2.append(score)
        for i in range(0,len(l2)-1):
            emo_score_list[l2[i]]=l2[len(l2)-1]
        l=fi.readline()
    return emo_score_list
#generate vector of ngrams in a text file. used in: mostFreqList
def ngramText(filename):
    textWords=[]
    f=open(filename,"r")
    line=f.readline()
    while line:
        textWords.extend(line.split())
        line=f.readline()
    f.close()
    return textWords
#returns a dictionary with word as key and freq as value from a list of words. used in: mostFreqList
def get_word_features(wordlist):
    wordlist = nltk.FreqDist(wordlist)
    result=[]
    for k in wordlist.keys():
        result.append([k,wordlist[k]])
    return result
#function that gets the k most frequent words from all the tweets, used in: buildUnigramVector
def mostFreqList(filename,k):
    d=get_word_features(ngramText(filename))
    #sorts list
    l=list(reversed(sorted(d, key=operator.itemgetter(1))))
    m=[w[0] for w in l[0:k]]
    return m
#function that returns the UnigramVector of positive and negative tweet files
def buildUnigramVector(positiveFile, negativeFile):
    positive=mostFreqList(positiveFile,3000)
    negative=mostFreqList(negativeFile,3000)
    total = positive + negative # total unigram vector
    for w in total:
        count = total.count(w)
        if (count > 1):
            while (count > 0):
                count = count - 1
                total.remove(w)
    # equalize unigrams sizes
    m = min([len(positive), len(negative)])
    return positive[0:m - 1], negative[0:m - 1]
#funtion that returns a dictionary of slag as key and real words as values aka translates slang to language
def loadSlangs(filename):
    slangs={}
    fi=open(filename,'r')
    line=fi.readline()
    while line:
        l=line.split(r',%,')
        if len(l) == 2:
            slangs[l[0]]=l[1][:-2]
        line=fi.readline()
    fi.close()
    return slangs


# globals:
#feature of words positive and negative values
words_weight = load_words_weight('data/words weights.txt')
#features of emoticons used in a tweet
emoticonDict= createEmoticonDictionary('data\emoticon.txt')
#unigram positive and negative
positive,negative = buildUnigramVector('data/training set/positive training tweets','data/training set/negative training tweets')
#slang to words
slangs=loadSlangs('data\internetSlangs.txt')

def mapTweet(line, words_weight, emoticonDict, positive, negative, slangs):
    pass


def loadMatrix(positive_tweets, negative_tweets, positive_label, negative_label):
    vectors = []
    labels = []
    f = open(positive_tweets, 'r')
    line = f.readline()
    while line:
        z = mapTweet(line, words_weight, emoticonDict, positive, negative, slangs)
        vectors.append(z)
      #  labels.append(float(poslabel))
        line = f.readline()
    f.close()


def main():
    # starts with knn
   # X, Y = loadMatrix('data/training set/positive training tweets.txt',
    #                  'data/training set/positive training tweets.txt', '1', '2')
    pass

main()