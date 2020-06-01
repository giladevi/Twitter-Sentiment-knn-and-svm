# TODO load tweets
# TODO load positive and negative tweets as training set
# TODO calculate X-vector of positive/negative values
# TODO label Y - vector of positive/negative labels
# TODO load testing set
# TODO calculate X-vector of positive/negative values for each tweet
# TODO label Y - using knn
# TODO label Y using svm

import re
import nltk
import operator
from sklearn import preprocessing as pr
from sklearn import svm
from sklearn import model_selection
import numpy as np

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
#function that gets the stop words list
def getStopWordList(stopWordListFileName):
    stopWords = []
    stopWords.append('at_user')
    stopWords.append('url')

    fp = open(stopWordListFileName, 'r')
    line = fp.readline()
    while line:
        word = line.strip()
        stopWords.append(word)
        line = fp.readline()
    fp.close()
    return stopWords

# globals:
#feature of words positive and negative values
words_weight = load_words_weight('data/words weights.txt')
#features of emoticons used in a tweet
emoticonDict= createEmoticonDictionary('data\emoticon.txt')
#unigram positive and negative
positive,negative = buildUnigramVector('data/training set/positive training tweets','data/training set/negative training tweets')
#slang to words
slangs=loadSlangs('data\internetSlangs.txt')
#stop words
stop_words=getStopWordList('data\stopWords.txt')

#look for 2 or more repetitions of character and replace with the character itself. used in: processTweet
def replaceTwoOrMore(s):
    pattern = re.compile(r"(.)\1{1,}", re.DOTALL)
    return pattern.sub(r"\1\1", s)
#function that replaces slangs to normal words used in: processTweet
def replaceSlangs(tweet,slangs):
    result=''
    words=tweet.split()
    for w in words:
        if w in slangs.keys():
            result=result+slangs[w]+" "
        else:
            result=result+w+" "
    return result
# arg tweet, stopWords list and internet slangs dictionary, Convert to lower case. used in mapTweet
def processTweet(tweet,stopWords,slangs):
    tweet = tweet.lower()
    #Convert www.* or https?://* to URL
    tweet = re.sub('((www\.[^\s]+)|(https?://[^\s]+))','url',tweet)
    tweet = re.sub('((www\.[^\s]+)|(http?://[^\s]+))','url',tweet)
    #Convert @username to AT_USER
    tweet = re.sub('@[^\s]+','at_user',tweet)
    #Remove additional white spaces
    tweet = re.sub('[\s]+', ' ', tweet)

 #   tweet = tweet.strip('\'"') # removing sepcial caracter
    processedTweet=replaceTwoOrMore(tweet) # replace multi-occurences by two
    words=replaceSlangs(processedTweet,slangs).split()
    processedTweet=''  # result variable
    for w in words:
        #strip punctuation
        if w in stopWords:
            None
        else:
#            w = w.strip('\'"%,.')
#            w=w.replace("'", "")
#            w=w.replace(".", "")
            w=w.replace('''"''', ''' ''')

        #ignore if it is a stop word
            processedTweet=processedTweet+w+' '
    return processedTweet
#function that returns the words weight feature as calculated in the paper. used in mapTweet
def afinnPolarity(tweet, words_weight):
    p=0.0
    nbr=0
    for w in tweet.split():
        if w in words_weight.keys():
            nbr+=1
            p+=words_weight[w]
    if (nbr != 0):
        return p/nbr
    else:
        return 0.0
# d for the emoticons dictionary, calculate the aggregate score of emoticons in a tweet. used in: mapTweet
def emoticonScore(tweet,d):
    s=0.0;
    l=tweet.split(" ")
    nbr=0;
    for i in range(0,len(l)):
        if l[i] in d.keys():
            nbr=nbr+1
            s=s+d[l[i]]
    if (nbr!=0):
        s=s/nbr
    return s
# returns list of hashtagged words in a tweet. used in: mapTweet
def hashtagWords(tweet):
    l=tweet.split()
    result=[]
    for w in tweet.split():
        if w[0]=='#' :
            result.append(w)

    return result
# returns 1 if there is uppercase words in tweet, 0 otherwise. used in mapTweet
def upperCase(tweet):
    result=0
    for w in tweet.split():
        if w.isupper():
            result=1
    return result
#returns 1 if there are exclamation signs. used in mapTweet
def exclamationTest(tweet):
    result=0
    if ("!" in tweet):
        result=1
    return result
#returns 1 if there are question signs. used in mapTweet
def questionTest(tweet):
    result=0
    if ("?" in tweet):
        result=1
    return result
# ratio of number of capitalized letters to the length of tweet. used in mapTweet
def freqCapital(tweet):
    count = 0
    for c in tweet:
        if (str(c).isupper()):
            count = count + 1
    if len(tweet) == 0:
        return 0
    else:
        return count / len(tweet)
#function that returns the unigram score of the tweet. used in: tweetMap
def scoreUnigram(tweet,posuni,neguni):
    pos=0
    neg=0
    l=len(tweet.split())
    for w in tweet.split():
        if w in posuni:
            pos+=1
        if w in neguni:
            neg+=1
    if (l!=0) :
        pos=pos/l
        neg=neg/l
    return [pos,neg]
#function that map the tweet by the features presented in the article
def mapTweet(tweet, words_weight, emoticonDict, positive, negative, slangs):
    out = []
    line = processTweet(tweet, stop_words, slangs)
    p = afinnPolarity(line, words_weight)
    out.append(p)
    out.append(float(emoticonScore(line, emoticonDict)))  # emo aggregate score be careful to modify weights
    out.append(float(len(hashtagWords(line)) / 40))  # number of hashtagged words
    out.append(float(len(line) / 140))  # for the length
    out.append(float(upperCase(line)))  # uppercase existence : 0 or 1
    out.append(float(exclamationTest(line)))
    out.append(float(line.count("!") / 140))
    out.append(float((questionTest(line))))
    out.append(float(line.count('?') / 140))
    out.append(float(freqCapital(line)))
    u = scoreUnigram(line, positive, negative)
    out.extend(u)
    return out
#this function returns the X (values) and Y (labels) of a group of positive and negative tweets
def loadMatrix(positive_tweets, negative_tweets, positive_label, negative_label):
    vectors = []
    labels = []
    f = open(positive_tweets, 'r')
    line = f.readline()
    while line:
        try:
            z = mapTweet(line, words_weight, emoticonDict, positive, negative, slangs)
            vectors.append(z)
            labels.append(float(positive_label))
        except:
            None
        line = f.readline()
    f.close()

    f = open(negative_tweets, 'r')
    line = f.readline()
    while line:
        try:
            z = mapTweet(line, words_weight, emoticonDict, positive, negative, slangs)
            vectors.append(z)
            labels.append(float(negative_label))
        except:
            None
        line = f.readline()
    #        print str(kneg)+"negative lines loaded : "+str(z)
    f.close()
    return vectors, labels
# map tweet into a vector

def trainModel(X,Y,knel,c): # relaxation parameter
    clf=svm.SVC(kernel=knel) # linear, poly, rbf, sigmoid, precomputed , see doc
    clf.fit(X,Y)
    return clf
# function to load test file in the csv format : sentiment,tweet
def loadTest(filename,scaler,normalizer):
    f=open(filename,'r')
    line=f.readline()
    labels=[]
    vectors=[]
    while line:
        l=line[:-1].split(r'","')
        s=float(l[0][1:])
        tweet=l[5][:-1]
        z=mapTweet(tweet,words_weight,emoticonDict,positive,negative,slangs)
        #z_scaled=scaler.transform(z)
        # z=normalizer.transform([z_scaled])
        # z=z[0].tolist()
        vectors.append(z)
        labels.append(s)
        line=f.readline()
    f.close()
    return vectors,labels
 # test a tweet against a built model
def predict(tweet,model):
    z=mapTweet(tweet,words_weight,emoticonDict,positive,negative,slangs) # mapping
    #z_scaled=scaler.transform(z)
    #z=normalizer.transform([z_scaled])
    #z=z[0].tolist()
    #return model.predict([z]).tolist()[0] # transform nympy array to list
    return model.predict([z]) # transform nympy array to list

def writeTest(filename,model): # function to load test file in the csv format : sentiment,tweet
    f=open(filename,'r')
    line=f.readline()
    fo=open(filename+".svm_result","w")
    fo.write("old,tweet,new\n")
    while line:
        l=line[:-1].split(r'","')
        s=float(l[0][1:])
        tweet=l[5][:-1]
        nl=predict(tweet,model)
        fo.write(r'"'+str(s)+r'","'+tweet+r'","'+str(nl)+r'"'+"\n")
        line=f.readline()
#        print str(kneg)+"negative lines loaded"
    f.close()
    fo.close()
    print("labelled test dataset is stores in : "+str(filename)+".svm_result")

def main():
    # X,Y,scales,normalizers
    X, Y = loadMatrix('data/training set/positive training tweets', 'data/training set/negative training tweets', 1, 0)
    # features standardization
    X_scaled = pr.scale(np.array(X))
    # to use later for testing data scaler.transform(X)
    scaler = pr.StandardScaler().fit(X)
    # features Normalization
    X_normalized = pr.normalize(X_scaled, norm='l2')  # l2 norm
    normalizer = pr.Normalizer().fit(X_scaled)  # as before normalizer.transform([[-1.,  1., 0.]]) for test
    # starts with svm
    X = X_normalized
    X = X.tolist()
    # 5 fold cross validation
    x = np.array(X)
    y = np.array(Y)
    KERNEL_FUNCTIONS = 'linear'
    C = [0.01 * i for i in range(1, 2)]
    ACC = 0.0
    PRE = 0.0
    iter = 0
    for knel in KERNEL_FUNCTIONS:
        for c in C:
            clf = svm.SVC(kernel=KERNEL_FUNCTIONS, C=c)
            #scores = model_selection.cross_validate(clf, x, y, cv=5, scoring='accuracy')
            # precisions = model_selection.cross_validate(clf, x, y, cv=5, scoring='precision')
            # if (scores.mean() > ACC and precisions.mean() > PRE):
            #     ACC = scores.mean()
            #     PRE = precisions.mean()
            #     KERNEL_FUNCTION = knel
            #     C_PARAMETER = c
            # iter = iter + 1
    MODEL = trainModel(X, Y, KERNEL_FUNCTIONS, 0.01)

    #V, L = loadTest('data/testing set/test_dataset.csv',scaler,normalizer)
    writeTest('data/testing set/test_dataset.csv', MODEL)
main()