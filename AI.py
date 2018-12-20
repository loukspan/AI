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
y_hiddenTest = df_test['GR']

#drops grammar
df_train.drop(df_test.columns[[0]], axis=1, inplace=True)
df_test.drop(df_test.columns[[0]], axis=1, inplace=True)

X_train = pd.get_dummies(df_train, columns=['SRC', 'CND'])
X_test = pd.get_dummies(df_test, columns=['SRC', 'CND'])

missing = set(X_train.columns)-set(X_test.columns)
for i in missing:
    X_test[i] = 0
X_test = X_test[X_train.columns]

from sklearn.tree import DecisionTreeClassifier

#Svc
dtc = DecisionTreeClassifier()
dtc.fit(X_train, y_train)
y_pred = dtc.predict(X_test)

y_hiddenTest = y_hiddenTest.values.tolist()
print(y_hiddenTest)
print(y_pred)

x = 0
sum = 0

for gr in y_pred:
    if gr == y_hiddenTest[x]:
        sum += 1
    x += 1

print(sum)
