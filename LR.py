import pandas as pd
import glob
import collections
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
import math


# Read mail text into a list
def read_text(path):
    txt = open(path)
    words = txt.read().split()
    words = [wrd for wrd in words if wrd != 'Subject:']
    txt.close()
    return words


# Split data set to train and test data sets
def split_data(data, split_ratio):
    msk = np.random.rand(len(data)) < split_ratio
    return [data[msk], data[~msk]]


# Make a list from a counter
def make_list(s, h):
    good = []
    bad = []
    for eachS in s:
        for eachH in h:
            if eachS[0] == eachH[0]:
                bad.append(eachS[0])
    print(good)
    print("BAD", bad)
    for eachS in s:
        if eachS[0] not in bad and eachS[0] not in good:
            good.append(eachS[0])
    for eachH in h:
        if eachH[0] not in bad and eachH[0] not in good:
            good.append(eachH[0])

    print(good)
    return good


# Get all texts into a list, find most common words and return a list with 100 top common words
def get_common_words(df):
    words = []
    s = []
    h = []
    for index, row in df.iterrows():
        if row['SPAM'] == 1:
            s.extend(row['TEXT'])
        else:
            h.extend(row['TEXT'])
        words.extend(row['TEXT'])
    #good = []
    #for each in collections.Counter(s).most_common(100) + collections.Counter(h).most_common(100):
    #    if each[0] not in good:
    #        good.append(each[0])

    #return good
    return make_list(collections.Counter(s).most_common(30), collections.Counter(h).most_common(30))  # , words)


# Create dataframe with all spam or ham mails with columns the common words
def make_spam_ham(columns, df_prmtr):
    all_columns = ['SPAM'] + columns
    df = pd.DataFrame(columns=all_columns)
    for index, row in df_prmtr.iterrows():
        row_list = [row['SPAM']]
        for word in columns:
            if word in row['TEXT']:
                row_list.append(1)
            else:
                row_list.append(0)
        df = df.append(pd.Series(row_list, index=all_columns), ignore_index=True)
    return df


def sigmoid(x):
    # Activation function used to map any real value between 0 and 1
    return 1 / (1 + np.exp(-x))


def net_input(weight, x, b):
    # Computes the weighted sum of inputs
    return np.dot(x, weight) + b


def probability(weight, x, b):
    # Returns the probability after passing through sigmoid
    return sigmoid(net_input(weight, x, b))


def overall_p(weight, x, spam, b):
    # Computes the cost function for all the training samples
    total_cost = np.sum(spam * np.log(probability(weight, x, b)) + (1 - spam) * np.log(1 - probability(weight, x, b)))
    return total_cost


# Start of the program
# Read all mails from all files except unused
# Create a dataframe with two columns
# one with text mail and
# one that has 1 if the mail is spam and 0 if the mail is ham
sp = 0
df_columns = ['TEXT', 'SPAM']
df_mails = pd.DataFrame(columns=df_columns)
pu_corp = glob.glob('pu_corpora_public*')
for pu in pu_corp:  # pu1 pu2 pu3 pua
    for part in glob.glob(pu + '/*'):   # part1 part2 ...
        for f in glob.glob(part + '/*'):
            for file_path in glob.glob(f + '/*.txt'):
                if 'unused' not in file_path:
                    text = read_text(file_path)
                    if 'spmsg' in file_path:
                        sp = 1
                    if 'legit' in file_path:
                        sp = 0
                    df_mails = df_mails.append(pd.Series([text, sp], index=df_columns), ignore_index=True)

# Split data set to train and test data sets
train, test = split_data(df_mails, 0.7)

print("Train dataframe \n", train)
print("Test dataframe \n", test)

# Save SPAM
y_train = train['SPAM']
y_train = y_train.astype('int')
y_test = test['SPAM']
print(y_train)

# Top 100 common words
common_words = get_common_words(train)
print("Common words in all train data set ", len(common_words), common_words)

# Create two dataframes one with all spam mails and one with all ham ones
X_train = make_spam_ham(common_words, train)
# X_test = make_spam_ham(common_words, test)
print("Train dataframe \n", X_train)
# print("Test dataframe \n", X_test)

weight = [1] * len(common_words)
b = 0
s_prev = 0
s = 0
j = 0
while True:
    print("LALALA")
    s_prev = s
    s = 0
    for index, row in X_train.iterrows():
        x = row.values.tolist()
        spam = x[0]
        x.remove(x[0])
        l_w = overall_p(weight, x, spam, b)
        s = s + (1 / l_w)
        prob = probability(weight, x, b)
        print(prob)
        for i, w in enumerate(weight):
            print("YOYO")
            w += 0.1 * (spam - prob) * x[i]
            b += 0.1 * (spam - prob)

        if index == 0:
            break
    j += 1
    if math.isclose(s_prev, s, rel_tol=1e-2) or j > 100:
        break

print(weight)


# Logistic Regression with only Departure Arrival at 0,31
# clf = LogisticRegression(solver='lbfgs')
# clf.fit(X_train, y_train)
# y_pred = clf.predict(X_test)

# Compare saved test SPAM column with prediction column
y_test = y_test.values.tolist()
y_test = [int(elem) for elem in y_test]
# print("Y_PRED: ", y_pred)
print("Y_TEST: ", y_test)

rate = 0
# for i, y in enumerate(y_pred):
#    if y == y_test[i]:
#        rate += 1

# rate = int((rate * 100) / len(y_pred))

# Print rate
print('We got ', rate, ' %')
