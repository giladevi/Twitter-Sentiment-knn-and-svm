import pandas as pd
from sklearn.utils import shuffle
import nltk
import operator
import re
import numpy as np
from sklearn import svm
from svm import SVM


def load_words_weight(filename):
    f = open(filename, 'r', encoding="ISO-8859-1")
    words_weights = {}
    line = f.readline()
    nbr = 0
    while line:
        nbr += 1
        ll = line[: -1].split('\t')
        words_weights[ll[0]] = float(ll[1])
        line = f.readline()

    return words_weights


def create_emoticon_dictionary(filename):
    emo_scores = {'Positive': 0.5, 'Extremely-Positive': 1.0, 'Negative': -0.5, 'Extremely-Negative': -1.0,
                  'Neutral': 0.0}
    emo_score_list = {}
    fi = open(filename, encoding="utf8")
    ll = fi.readline()
    while ll:
        ll = ll.replace("\xc2\xa0", " ")
        li = ll.split(" ")
        l2 = li[:-1]
        l2.append(li[len(li) - 1].split("\t")[0])
        sentiment = li[len(li) - 1].split("\t")[1][:-1]
        score = emo_scores[sentiment]
        l2.append(score)
        for i in range(0, len(l2) - 1):
            emo_score_list[l2[i]] = l2[len(l2) - 1]
        ll = fi.readline()
    return emo_score_list


def get_word_features(word_list):
    word_list = nltk.FreqDist(word_list)
    result = []
    for k in word_list.keys():
        result.append([k, word_list[k]])
    return result


def ngram_text(filename):
    text_words = []
    f = open(filename, "r", encoding="ISO-8859-1")
    line = f.readline()
    while line:
        text_words.extend(line.split())
        line = f.readline()
    f.close()
    return text_words


def most_freq_list(filename, k):
    d = get_word_features(ngram_text(filename))
    ll = list(reversed(sorted(d, key=operator.itemgetter(1))))
    m = [w[0] for w in ll[0:k]]
    return m


def load_slangs(filename):
    local_slangs = {}
    fi = open(filename, 'r', encoding="ISO-8859-1")
    line = fi.readline()
    while line:
        ll = line.split(r',%,')
        if len(ll) == 2:
            local_slangs[ll[0]] = ll[1][:-2]
        line = fi.readline()
    fi.close()
    return local_slangs


def get_stop_word_list(stop_word_list_file_name):
    local_stop_words = ['at_user', 'url']
    fp = open(stop_word_list_file_name, 'r', encoding="ISO-8859-1")
    line = fp.readline()
    while line:
        word = line.strip()
        local_stop_words.append(word)
        line = fp.readline()
    fp.close()
    return local_stop_words


def replace_two_or_more(s):
    pattern = re.compile(r"(.)\1{1,}", re.DOTALL)
    return pattern.sub(r"\1\1", s)


# function that replaces slangs to normal words used in: processTweet
def replace_slangs(tweet, slangs):
    result = ''
    words = tweet.split()
    for w in words:
        if w in slangs.keys():
            result = result + slangs[w] + " "
        else:
            result = result + w + " "
    return result


# arg tweet, stopWords list and internet slangs dictionary, Convert to lower case. used in mapTweet
def process_tweet(tweet, stop_words, slangs):
    tweet = tweet.lower()
    # Convert www.* or https?://* to URL
    tweet = re.sub('((www\.[^\s]+)|(https?://[^\s]+))', 'url', tweet)
    tweet = re.sub('((www\.[^\s]+)|(http?://[^\s]+))', 'url', tweet)
    # Convert @username to AT_USER
    tweet = re.sub('@[^\s]+', 'at_user', tweet)
    # Remove additional white spaces
    tweet = re.sub('[\s]+', ' ', tweet)

    #   tweet = tweet.strip('\'"') # removing sepcial caracter
    processed_tweet = replace_two_or_more(tweet)  # replace multi-occurences by two
    words = replace_slangs(processed_tweet, slangs).split()
    processed_tweet = ''  # result variable
    for w in words:
        # strip punctuation
        if w not in stop_words:
            w = w.replace('''"''', ''' ''')

            # ignore if it is a stop word
            processed_tweet = processed_tweet + w + ' '
    return processed_tweet


def afinn_polarity(tweet, local_words_weight):
    p = 0.0
    nbr = 0
    for w in tweet.split():
        if w in local_words_weight.keys():
            nbr += 1
            p += local_words_weight[w]
    if nbr != 0:
        return p / nbr
    else:
        return 0.0


def emoticon_score(tweet, d):
    s = 0.0
    ll = tweet.split(" ")
    nbr = 0
    for i in range(0, len(ll)):
        if ll[i] in d.keys():
            nbr = nbr + 1
            s = s + d[ll[i]]
    if nbr != 0:
        s = s / nbr
    return s


def hashtag_words(tweet):
    result = []
    for w in tweet.split():
        if w[0] == '#':
            result.append(w)
    return result


def upper_case(tweet):
    result = 0
    for w in tweet.split():
        if w.isupper():
            result = 1
    return result


def question_test(tweet):
    result = 0
    if "?" in tweet:
        result = 1
    return result


def freqCapital(tweet):
    count = 0
    for c in tweet:
        if (str(c).isupper()):
            count = count + 1
    if len(tweet) == 0:
        return 0
    else:
        return count / len(tweet)


def map_tweet(tweet, local_words_weight, local_emoticon_dict, local_slangs, stop_words):
    out = []
    line = process_tweet(tweet, stop_words, local_slangs)
    p = afinn_polarity(line, local_words_weight)
    out.append(p)
    out.append(float(emoticon_score(line, local_emoticon_dict)))  # emo aggregate score be careful to modify weights
    #out.append(float(len(hashtag_words(line)) / 140))  # number of hashtagged words
    #out.append(float(len(line) / 140))  # for the length
    out.append(float(line.count("!") / 140))
    out.append(float(line.count('?') / 140))
    out.append(float(freqCapital(line)))
    return out


# def load_matrix2(dataset_path, train_number, test_number, percent):
#     words_weight = load_words_weight('data/words weights.txt')
#     emoticon_dict = create_emoticon_dictionary('data/emoticon.txt')
#     positive, negative = build_unigram_vector('data/training set/positive training tweets',
#                                               'data/training set/negative training tweets')
#     slangs = load_slangs('data/internetSlangs.txt')
#     stop_words = get_stop_word_list('data/stopWords.txt')
#
#     dataset = pd.read_csv(dataset_path, encoding="ISO-8859-1", header=None)
#
#     dataset = shuffle(dataset)
#
#     train_set_pos = dataset[dataset[0] == 4].head(int(train_number * percent / 100))
#     train_set_neg = dataset[dataset[0] == 0].head(train_number - int(train_number * percent / 100))
#     test_set_pos = dataset[dataset[0] == 4].tail(int(test_number * percent / 100))
#     test_set_neg = dataset[dataset[0] == 0].tail(test_number - int(test_number * percent / 100))
#
#     train_set = shuffle(pd.concat([train_set_pos, train_set_neg]))
#     test_set = shuffle(pd.concat([test_set_pos, test_set_neg]))
#
#     train_tweets = train_set.iloc[:, 5].values
#     train_y = np.subtract(np.divide(train_set.iloc[:, 0].values, 2), 1)
#
#     test_tweets = test_set.iloc[:, 5].values
#     test_y = np.subtract(np.divide(test_set.iloc[:, 0].values, 2), 1)
#
#     train_x = np.array([map_tweet(tweet, words_weight, emoticon_dict, positive, negative, slangs, stop_words)
#                         for tweet in train_tweets])
#
#     test_x = np.array([map_tweet(tweet, words_weight, emoticon_dict, positive, negative, slangs, stop_words)
#                        for tweet in test_tweets])
#
#     return train_x, train_y, test_x, test_y


def load_matrix(dataset_path, train_number, test_number):
    words_weight = load_words_weight('data/words weights.txt')
    emoticon_dict = create_emoticon_dictionary('data/emoticon.txt')
    slangs = load_slangs('data/internetSlangs.txt')
    stop_words = get_stop_word_list('data/stopWords.txt')

    dataset = pd.read_csv(dataset_path, encoding="ISO-8859-1", header=None)

    dataset = shuffle(dataset)

    train_set = dataset[0: train_number]
    test_set = dataset[train_number: train_number + test_number]

    train_tweets = train_set.iloc[:, 5].values
    train_y = np.subtract(np.divide(train_set.iloc[:, 0].values, 2), 1)

    test_tweets = test_set.iloc[:, 5].values
    test_y = np.subtract(np.divide(test_set.iloc[:, 0].values, 2), 1)

    train_x = np.array([map_tweet(tweet, words_weight, emoticon_dict, slangs, stop_words)
                        for tweet in train_tweets])

    test_x = np.array([map_tweet(tweet, words_weight, emoticon_dict, slangs, stop_words)
                       for tweet in test_tweets])

    return train_x, train_y, test_x, test_y


def cross_validation(x, y, folds_number):
    avg = 0
    precision = 0
    recall = 0
    fold_size = int(y.shape[0] / folds_number)

    for i in range(folds_number):
        train_x = np.concatenate((x[(i - 1) * fold_size: i * fold_size], x[(i + 1) * fold_size: (i + 2) * fold_size]))
        train_y = np.concatenate((y[(i - 1) * fold_size: i * fold_size], y[(i + 1) * fold_size: (i + 2) * fold_size]))

        test_x = x[i * fold_size: (i + 1) * fold_size]
        test_y = y[i * fold_size: (i + 1) * fold_size]

        clf = SVM()
        clf.fit(train_x, train_y)
        y_pred = clf.predict(test_x)
        avg += np.average(y_pred == test_y)
        precision += np.sum(np.bitwise_and(y_pred == 1, test_y == 1)) / np.sum(y_pred == 1)
        recall += np.sum(np.bitwise_and(y_pred == 1, test_y == 1)) / np.sum(test_y == 1)

    return avg / folds_number * 100, precision / folds_number, recall / folds_number


def test():
    dataset = pd.read_csv("data/data.csv", encoding="ISO-8859-1", header=None)
    print(dataset[dataset[0] == 4].head(5))


def main():
    # train_x, train_y, test_x, test_y = load_matrix2("data/data.csv", 1000, 1000, 49.1)
    train_x, train_y, test_x, test_y = load_matrix("data/sapirData.csv", 1100, 20)

    print('num of 1:', np.sum(test_y == 1))

    #clf = svm.SVC(kernel='linear', C=0.01)
    clf = SVM()

    clf.fit(train_x, train_y)
    y_pred = clf.predict(test_x)
    res = np.average(y_pred == test_y) * 100

    print('simple accurate: {:.2f}%'.format(res))
    print('simple precision: {:.2f}'.format(np.sum(np.bitwise_and(y_pred == 1, test_y == 1)) / np.sum(y_pred == 1)))
    print('simple recall: {:.2f}'.format(np.sum(np.bitwise_and(y_pred == 1, test_y == 1)) / np.sum(test_y == 1)))

    accurate, precision, recall = cross_validation(train_x, train_y, 5)
    print('cross validation accurate: {:.2f}%\ncross validation precision: {:.2f}\ncross validation recall: {:.2f}'
          .format(accurate, precision, recall))


# def test():
#     y_pred = np.array([1, -1, 1, -1])
#     test_y = np.array([1, 1, -1, 1])
#
#     print(np.sum(np.bitwise_and(y_pred == 1, test_y == 1)) / np.sum(y_pred == 1))
#     print(np.sum(np.bitwise_and(y_pred == 1, test_y == 1)) / np.sum(test_y == 1))


if __name__ == '__main__':
    main()
