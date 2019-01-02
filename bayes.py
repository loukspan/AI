import pandas as pd
# from sklearn.naive_bayes import GaussianNB
import glob
import collections
import numpy as np


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
                #if abs(eachS[1] - eachH[1]) > min(eachS[1], eachH[1]):
                #    good.append(eachS[0])
                #else:
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
    #common_words_list = []
    #for c in most_common:
     #   common_words_list.append(c[0])
    #return common_words_list
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
    return make_list(collections.Counter(s).most_common(500), collections.Counter(h).most_common(500))  # , words)


def calculate_important_words(common, p_s, p_h):
    common_temp = []
    p_s_temp = []
    p_h_temp = []
    for index, c in enumerate(common):
        if abs(p_s[index] - p_h[index]) > 0.2:
            common_temp.append(c)
            p_s_temp.append(p_s[index])
            p_h_temp.append(p_h[index])
    return [common_temp, p_s_temp, p_h_temp]


# Create dataframe with all spam or ham mails with columns the common words
def make_spam_ham(columns, cl, df_prmtr):
    df = pd.DataFrame(columns=columns)
    for index, row in df_prmtr.iterrows():
        row_list = []
        if row['SPAM'] == cl:
            for word in columns:
                if word in row['TEXT']:
                    row_list.append(1)
                else:
                    row_list.append(0)
            df = df.append(pd.Series(row_list, index=columns), ignore_index=True)
    return df


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
y_test = test['SPAM']

# Top 100 common words
common_words = get_common_words(train)
print("Common words in all train data set ", len(common_words), common_words)

# Create two dataframes one with all spam mails and one with all ham ones
df_spam = make_spam_ham(common_words, 1, train)
df_ham = make_spam_ham(common_words, 0, train)
print("Train Spam dataframe \n", df_spam)
print("Train Ham dataframe \n", df_ham)


# Calculate Probabilities

# Number of spam and ham mails
nOfSpam = train['SPAM'][train['SPAM'] == 1].count()
nOfHam = train['SPAM'][train['SPAM'] == 0].count()
print("Number of observations in spam and ham mails: ", nOfSpam, nOfHam)

# Number of total mails
nOfMails = train['SPAM'].count()
print("Number of total mails: ", nOfMails)

# Probability of being a mail spam or ham
P_spam = nOfSpam/nOfMails
P_ham = nOfHam/nOfMails
print("Probability of being a mail spam or ham: ", P_spam, P_ham)

# Count of each word given Spam
counts = df_spam.apply(pd.value_counts)
counts = counts.fillna(0)
print(counts)
nOfWordsSpam = counts.iloc[1].values.tolist()
print(nOfWordsSpam)

# Probability of each word given Spam
P_wordSpam = []
for nOfwS in nOfWordsSpam:
    P_wordSpam.append(nOfwS/nOfSpam)
print("Probability of each word given Spam: ", P_wordSpam)

# Count of each word given Ham
counts = df_ham.apply(pd.value_counts)
counts = counts.fillna(0)
print(counts)
nOfWordsHam = counts.iloc[1].values.tolist()
print(nOfWordsHam)

# Probability of each word given Ham
P_wordHam = []
for nOfwH in nOfWordsHam:
    P_wordHam.append(nOfwH/nOfHam)
print("Probability of each word given Ham: ", P_wordHam)

# common_words, P_wordSpam, P_wordHam = calculate_important_words(common_words, P_wordSpam, P_wordHam)
print(len(common_words), common_words)
print(len(P_wordSpam), P_wordSpam)
print(len(P_wordHam), P_wordHam)


# For each mail in test data set find its probability to be spam or ham and update y_pred (prediction list)
y_pred = []
for i, mail in test.iterrows():
    # Initially each probability is probability of being spam or ham
    ProbGivenSpam = P_spam
    ProbGivenHam = P_ham
    for j, w in enumerate(common_words):
        # If text of a mail contains a word in top 100 common words
        if w in mail['TEXT']:
            # Multiply the word's probability with the rest
            ProbGivenSpam *= P_wordSpam[j]
            ProbGivenHam *= P_wordHam[j]
    # Update prediction list with 1 if it's more possible to be spam
    if ProbGivenSpam > ProbGivenHam:
        y_pred.append(1)
    # and with 0 if it's more possible to be ham
    else:
        y_pred.append(0)

# train.drop(test.columns[[0]], axis=1, inplace=True)
# gnb = GaussianNB()
# gnb.fit(X_train, y_train)
# y_pred = gnb.predict(X_test)

# Compare saved test SPAM column with prediction column
y_test = y_test.values.tolist()
y_test = [int(elem) for elem in y_test]
print("Y_PRED: ", y_pred)
print("Y_TEST: ", y_test)

rate = 0
for i, y in enumerate(y_pred):
    if y == y_test[i]:
        rate += 1

rate = int((rate * 100) / len(y_pred))

# Print rate
print('We got ', rate, ' %')
