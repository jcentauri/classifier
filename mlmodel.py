import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
import pickle

df = pd.read_csv('bshows.csv')

#use required features
cdf = df[['Age','Experience','Nationality','Go']]
print(cdf)
#Training Data and Predictor Variable
# Use all data for training (tarin-test-split not used)

X = cdf[['Age','Experience','Nationality']]
y = cdf['Go']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=9) # 80% training and 20% test
print(X_train)
print(X_test)
print(y_train)
print(y_test)


regressor = LogisticRegression()

#Fitting model with trainig data
regressor.fit(X_train,y_train)

# predict the label on the traning data
predict_train = regressor.predict(X_train)
print(predict_train)
# predict the model on the test data
y_pred = regressor.predict(X_test)
print(y_pred)

print("Accuracy:", metrics.accuracy_score(y_test, y_pred))

print("Confusion_Matrix:\n", confusion_matrix(y_test, y_pred) )

print("F1_Score:", f1_score(y_test, y_pred))

print("Precision_Score:", precision_score(y_test, y_pred))

print("Recall_Score:", recall_score(y_test, y_pred))

file_name='my_file.pkl'
f = open(file_name,'wb')
pickle.dump(regressor,f)
f.close()