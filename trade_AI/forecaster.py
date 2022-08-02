import pandas as pd

from sklearn.ensemble import RandomForestClassifier as rfc
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

df = pd.read_csv('df.csv').dropna()

train = df.iloc[:-4000,:]
val = df.iloc[-4000:-2000,:]
test = df.iloc[-2000:,:]
x_train = train.iloc[:, :-1]
y_train = train.iloc[:, -1]
# x_val = val[:][:-1]
# y_val = val[:][-1]
x_test = test.iloc[:, :-1]
y_test = test.iloc[:, -1]



# model = rfc(class_weight={0:0.01,1:100})
# model.fit(x_train, y_train)
# y_pred = model.predict(x_test)
# print(accuracy_score(y_test, y_pred))
# print('===========')
# print(confusion_matrix(y_test, y_pred))
