```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

z = pd.read_csv('house_price.txt', sep = ',')
```


```python
z.loc[:, ("y1")] = (z['price'] > z['price'].median()).astype(int)
z = z.drop(['index','price','sq_price'], axis = 1)
z_low = z.loc[z['y1'] == 0][['area','bathrooms','y1']]
z_high = z.loc[z['y1'] == 1][['area','bathrooms','y1']]
```


```python
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
trainX, testX, trainY, testY = train_test_split(z[['area','bathrooms']], z['y1'], test_size = 0.25, random_state = 33)
ss = StandardScaler()
trainX_std = ss.fit_transform(trainX)
testX_std = ss.transform(testX)

```


```python
## l2-regularized logistic model
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(solver = 'liblinear').fit(trainX_std, trainY)
```


```python
from sklearn.metrics import confusion_matrix, classification_report
predictY = lr.predict(testX_std)
CFmat = confusion_matrix(y_true = testY, y_pred = predictY)
print(CFmat)
print(classification_report(testY, predictY))
```


```python
#Linear SVM
from sklearn.svm import LinearSVC
lsvc = LinearSVC().fit(trainX_std, trainY)
predictY = lsvc.predict(testX_std)
from sklearn.metrics import classification_report
print(classification_report(testY, predictY))
```
    


```python
#kernel SVM
from sklearn.svm import SVC
ksvm = SVC(kernel = 'rbf', C=1, gamma=0.1).fit(trainX_std,trainY)
predictY2 = ksvm.predict(testX_std)
print(classification_report(testY,predictY2))
```



```python
#kNN
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
accuracy_rate = []
n_train = len(trainY)
for i in range(1, n_train + 1):
    knn = KNeighborsClassifier(n_neighbors = i).fit(trainX_std, trainY)
    pred_i = knn.predict(testX_std)
    accuracy_rate.append(accuracy_score(testY, pred_i))

plt.figure(figsize=(10,6))
plt.plot(range(1, n_train+1), accuracy_rate, color='blue',linestyle='dashed',marker='o',markerfacecolor='red',markersize=10)
plt.xlabel('k')
plt.ylabel('Accuracy Rate')
print('Max accuracy:', max(accuracy_rate), 'at k =', accuracy_rate.index(max(accuracy_rate))+1)
```



```python
## Decision Tree
from sklearn.tree import DecisionTreeClassifier
dtree = DecisionTreeClassifier().fit(trainX_std,trainY)
```


```python
from sklearn.metrics import confusion_matrix, classification_report
predictDt = dtree.predict(testX_std)
CFmat = confusion_matrix(y_true = testY, y_pred = predictDt)
print(CFmat)
print(classification_report(testY, predictDc))
```


```python
from sklearn import tree
fig = plt.figure(figsize = (25,20))
_ = tree.plot_tree(dtree,
                  feature_names = trainX_std.columns,
                  class_names = ['low','high'],
                  filled = True)
```


```python
from sklearn.tree import export_graphviz
export_graphviz(dtree,
               out_file = 'tree.dot',
               class_names = ['low','high'],
               feature_names = ['area','bathrooms'],
               impurity = False,
               filled = True)
import graphviz
with open('tree.dot') as f:
    dotgraph = f.read()
    
```


```python
## naive Bayes
from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB().fit(trainX_std,trainY)
```


```python
from sklearn.metrics import confusion_matrix, classification_report
predictYgnb = gnb.predict(testX_std)
CFmat = confusion_matrix(y_true = testY, y_pred = predictYgnb)
print(CFmat)
print(classification_report(testY,predictYgnb))
```
