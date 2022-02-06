> ### Input data

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

z = pd.read_csv('house_price.txt', sep = ',')
z.loc[:, ("y1")] = (z['price'] > z['price'].median()).astype(int)
z = z.drop(['index','price','sq_price'], axis = 1)
z_low = z.loc[z['y1'] == 0][['area','bathrooms','y1']]
z_high = z.loc[z['y1'] == 1][['area','bathrooms','y1']]
```

> ### Split data and normalization 
```python
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
trainX, testX, trainY, testY = train_test_split(z[['area','bathrooms']], z['y1'], test_size = 0.25, random_state = 33)
ss = StandardScaler()
trainX_std = ss.fit_transform(trainX)
testX_std = ss.transform(testX) # Use the mean and sd of trainX to standardize testX to make sure that testX and trainX are in the same scale after transformation.
```

> ### l2-regularized(default) logistic model
```python
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(solver = 'liblinear').fit(trainX_std, trainY)
#可选参数
LogisticRegression(solver = 'lbfgs', penalty = 'l2', C = 1)
```
* 'lbfgs' is the default solver and it's often used for multi-class classification; 
* 'liblinear' is a good choice for small datasets but it can only be used for 2-class classification
* penalty can be 'l1' or 'l2', the default value is 'l2'

> ### Linear SVM
```python
from sklearn.svm import LinearSVC
lsvc = LinearSVC().fit(trainX_std, trainY)
predictY = lsvc.predict(testX_std)
```

> ### kernel SVM
```python
from sklearn.svm import SVC
ksvm = SVC(kernel = 'rbf', C=1, gamma=0.1).fit(trainX_std,trainY)
predictY2 = ksvm.predict(testX_std)
```

> ### kNN——choose best k
```python
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
> ### Decision Tree

```python
from sklearn.tree import DecisionTreeClassifier
dtree = DecisionTreeClassifier().fit(trainX_std,trainY)
```


> ### Naive Bayes

```python
from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB().fit(trainX_std,trainY)
```
> ### ROC curve and AUC
```
from sklearn.metrics import roc_curve, auc
fpr, tpr, thresholds = roc_curve(testY, lr.predict_proba(testX_std)[:,1])
roc_auc = auc(fpr, tpr)
plt.title('ROC Curve')
plt.plot(fpr, tpr, marker = 'o', label = 'AUC = %0.3f' % roc_auc)
plt.legend(loc = 'best')
plt.plot([0,1], [0,1], 'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.show()
```
> ### Confusion Matrix and Classification Report 
```python
from sklearn.metrics import confusion_matrix, classification_report
predictYgnb = gnb.predict(testX_std)
CFmat = confusion_matrix(y_true = testY, y_pred = predictYgnb)
print(CFmat)
print(classification_report(testY,predictYgnb))
```
