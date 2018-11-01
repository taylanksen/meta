#!/usr/bin/env python3

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
import time
import logging
logging.basicConfig(level=logging.DEBUG)
log = logging.getLogger(__name__)
'''
If you want to set the logging level from a command-line option such as:
  --log=INFO
'''

class Meta(LogisticRegression):
    """
    
    
    """
    
    def __init__(s, *args, **kwargs):
        # just call LogisticRegression init passing args straight through
        s.div_feats = None
        s.meta_feat = None
        super().__init__(*args, **kwargs)
        
        
    def create_augmented_X(s, X: pd.DataFrame) -> pd.DataFrame:
        """ """
        if s.div_feats and s.meta_feat:
            non_div = list(set(X.columns) - set(s.div_feats))
            newX = X[non_div].copy()
            #new_feats = product(s.div_feats, ['_m','_f'])
            for feat in s.div_feats:
                assert(feat in X.columns)
                male_feat = feat + '_m'
                female_feat = feat + '_f'
                newX[male_feat] = X[feat]*X[s.meta_feat]
                newX[female_feat] = X[feat]*(X[s.meta_feat] == 0)
            return newX
        else:
            return X
            
        
    def fit(s, 
            X:pd.DataFrame, 
            y:pd.Series, 
            div_feats:list=None, 
            meta_feat:str=None, 
            sample_weight=None):
        
        s.div_feats = div_feats
        s.meta_feat = meta_feat
        if div_feats is None:
            log.warn('no divergent features specified, use: ')
            log.warn('  fit(X, y, div_feats=<>, meta_feat=<>)')

        if meta_feat is None:
            log.warn('no meta feature specified, use: ')
            log.warn('  fit(X, y, div_feats=<>, meta_feat=<>)')
        
        #non_div = list(set(X.columns) - set(div_feats) - set(meta_feat))
        # create divergent features
        newX = s.create_augmented_X(X)
        super().fit(newX, y, sample_weight)
    
    
    def predict(s, X:pd.DataFrame):
        newX = s.create_augmented_X(X)
        return super().predict(newX)
    
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
    X = df[['AU06_r','AU12_r', 'AU07_r', 'is_male']]
    y = df['y']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5)
    
    clf = LogisticRegression(penalty='l1', C=1)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    acc = metrics.accuracy_score(y_pred,y_test)
    print('LR:  acc %.4f' % acc)

    clf = Meta(penalty='l1', C=1)
    clf.fit(X_train, y_train, div_feats=['AU06_r', 'AU12_r'], meta_feat='is_male')
    y_pred = clf.predict(X_test)
    acc = metrics.accuracy_score(y_pred,y_test)
    print('Meta: acc %.4f' % acc)
    print('meta feature: ', clf.meta_feat)
    print('div features: ', clf.div_feats)
    
if __name__ == '__main__':
    print('starting Meta test')
    test()
    print('Meta test complete')
