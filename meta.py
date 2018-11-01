#!/usr/bin/env python3

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
import time


class Meta(LogisticRegression):
    def __init__(s, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
    def fit(s, X, y, sample_weight=None):
        super().fit(X, y, sample_weight)
    
    
'''
    def __init__(self, penalty='l2', dual=False, tol=1e-4, C=1.0,
                 fit_intercept=True, intercept_scaling=1, class_weight=None,
                 random_state=None, solver='liblinear', max_iter=100,
                 multi_class='ovr', verbose=0, warm_start=False, n_jobs=1):
        linear_model.__init__(penalty=penalty, dual=dual, tol=tol, C=C, 
            fit_intercept=fit_intercept, intercept_scaling=intercept_scaling, 
            class_weight=class_weight, random_state=random_state, solver=solver, 
            max_iter=max_iter, multi_class=multi_class, verbose=verbose, 
            warm_start=warm_start, n_jobs=1)
'''
def test():
    df = pd.read_csv('test/hs_avg.csv', skipinitialspace=True)
    df.rename(columns={'truth':'y'}, inplace=True)
    X = df[['AU06_r','AU12_r']]
    y = df['y']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5)
    
    for name, model in zip(['lr', 'meta'],[LogisticRegression, Meta]):
        clf = model(penalty='l1', C=1)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        acc = metrics.accuracy_score(y_pred,y_test)
        print('%s acc %.4f' % (name, acc))
        
    
if __name__ == '__main__':
    print('starting Meta test')
    test()
    print('Meta test complete')
