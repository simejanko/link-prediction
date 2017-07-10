import numpy as np
import os
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score

test_auc = 0
valid_auc = 0
n = int(len(os.listdir('data/npy'))/4)
n = 2
for i in range(n):
    X_train = np.load('data/npy/X_train_{}.npy'.format(i))
    X_test = np.load('data/npy/X_test_{}.npy'.format(i))
    y_train = np.load('data/npy/y_train_{}.npy'.format(i))
    y_test = np.load('data/npy/y_test_{}.npy'.format(i))

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    #lr = LogisticRegression(penalty='l2', C=1)
    #lr = RandomForestClassifier(n_estimators=300, max_depth=12, min_samples_split=50)

    #lr = SVC(C=1, kernel='rbf', gamma=0.005, probability=True, cache_size=1500)
    #valid_auc += cross_val_score(lr, X_train, y_train, cv=5, scoring='roc_auc').mean()/n
    lr.fit(X_train, y_train)
    y_pred = lr.predict_proba(X_test)[:, 1]
    test_auc += roc_auc_score(y_test, y_pred) / n

#print('avg. CV AUC: {}'.format(valid_auc))
print('avg. test AUC: {}'.format(test_auc))

