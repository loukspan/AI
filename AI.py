import xml.etree.cElementTree as et
import pandas as pd

#read train_instances.xml
parsed_train = et.parse('dataset/train_instances.xml')
df_columns = ['GR', 'MP', 'PQ', 'SRC', 'CND']
df_train = pd.DataFrame(columns=df_columns)

for pair in parsed_train.getroot():
    gr = pair.get('GR')
    mp = pair.get('MP')
    pq = pair.get('PQ')
    src = pair.find('sourceSentence').text
    cnd = pair.find('candidateParaphrase').text

    df_train = df_train.append(
        pd.Series([gr, mp, pq, src, cnd], index=df_columns), ignore_index=True)

#read test_instances.xml
parsed_test = et.parse('dataset/test_instances.xml')
df_columns = ['GR', 'MP', 'PQ', 'SRC', 'CND']
df_test = pd.DataFrame(columns=df_columns)

for pair in parsed_test.getroot():
    gr = pair.get('GR')
    mp = pair.get('MP')
    pq = pair.get('PQ')
    src = pair.find('sourceSentence').text
    cnd = pair.find('candidateParaphrase').text

    df_test = df_test.append(
        pd.Series([gr, mp, pq, src, cnd], index=df_columns), ignore_index=True)

y_train = df_train[['GR']]
y_a = df_train['GR']

#drops grammar
df_train.drop(df_test.columns[[0]], axis=1, inplace=True)
df_test.drop(df_test.columns[[0]], axis=1, inplace=True)

X_train = pd.get_dummies(df_train, columns=['SRC', 'CND'])
X_test = pd.get_dummies(df_test, columns=['SRC', 'CND'])

missing = set(X_train.columns)-set(X_test.columns)
for i in missing:
    X_test[i] = 0
X_test = X_test[X_train.columns]

print(X_test.shape)
print(X_train.shape)

from sklearn.tree import DecisionTreeClassifier

#Svc
dtc = DecisionTreeClassifier()
dtc.fit(X_train, y_train)
y_pred = dtc.predict(X_test)

y_a = y_a.values.tolist()
print(y_a)
print(y_pred)

x = 0
for item in y_a:
    for item1 in y_pred:
        if item == item1:
            x += 1
print(x)
