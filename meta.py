#!/usr/bin/env python3
"""
This file contains the implementation of the metamorphic model which is backbone
of the Gender Augmented Linear (GAL) and CHIMERA models.
"""
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
    Implementation of the metamorphic model which is logistic regression with a 
    partial polynomial (degree 2) extension of the features. 
        given: a metamorphic feature (such as gender) and 
               a set of divergent features, 
        replaces each divergent feature with two features, one multiplied by
        the metamorphic feature, and the other multiplied by 
        (1 - metamoprhic feature).

    Implemented as an overloaded extension of sklearn's LogisticRegression.
    ----------------------------------------------------------------------------
    modified public method:
    
    fit(X:pd.DataFrame,      # input features in columns 
        y:pd.Series,         # output (should be 0/1)
        sample_weight=None,  
        div_feats:list=None, # list of str, column names of divergent features
        meta_feat:str=None)  # the column name of the feature which will 
    
    
    """
    
    def __init__(s, *args, **kwargs):
        """ just call LogisticRegression init passing args straight through.
            def __init__(self, penalty='l2', dual=False, tol=1e-4, C=1.0,
                         fit_intercept=True, intercept_scaling=1, 
                         class_weight=None,random_state=None, 
                         solver='liblinear', max_iter=100, multi_class='ovr', 
                         verbose=0, warm_start=False, n_jobs=1):
        """        
        s._div_feats = None
        s._meta_feat = None
        super().__init__(*args, **kwargs)
            
        
    def fit(s, X:pd.DataFrame, 
            y:pd.Series,  
            sample_weight=None,
            div_feats:list=None, 
            meta_feat:str=None):
        """Fit the model according to the given training data.

        Parameters
        ----------
        X : pd.DataFrame, shape (n_samples, n_features)
            Training vector, where n_samples is the number of samples and
            n_features is the number of features.

        y : pd.Series, shape (n_samples,)
            Target vector relative to X.

        sample_weight : array-like, shape (n_samples,) optional
            Array of weights that are assigned to individual samples.
            If not provided, then each sample is given unit weight.
        
        meta_feat : str, column name of feature which is believed to be 
            metamorphic (i.e. modifies the divergent features). The 
            metamorphic feature data should either be bool or in the
            range [0,1].

        div_feats : list, column names of features to be treated as divergent.
            Divergent features are replaced with two features, one multiplied
            by the meta feature, the other multiplied by (1 - meta feature)

        Returns
        -------
        self : object
            Returns self.
        """        
        
        s._div_feats = div_feats
        s._meta_feat = meta_feat
        if div_feats is None:
            log.warn('no divergent features specified, use: ')
            log.warn('  fit(X, y, div_feats=<>, meta_feat=<>)')

        if meta_feat is None:
            log.warn('no meta feature specified, use: ')
            log.warn('  fit(X, y, div_feats=<>, meta_feat=<>)')
        
        #non_div = list(set(X.columns) - set(div_feats) - set(meta_feat))
        # create divergent features
        newX = s._create_augmented_X(X)
        super().fit(newX, y, sample_weight)
    
        
    def decision_function(s, X:pd.DataFrame):
        """Predict confidence scores for samples.

        The confidence score for a sample is the signed distance of that
        sample to the hyperplane.

        Parameters
        ----------
        X : pd.DataFrame, shape = (n_samples, n_features)
            Samples.

        Returns
        -------
        array, shape=(n_samples,) if n_classes == 2 else (n_samples, n_classes)
            Confidence scores per (sample, class) combination. In the binary
            case, confidence score for self.classes_[1] where >0 means this
            class would be predicted.
        """        
        newX = s._create_augmented_X(X)
        return super().decision_function(newX)        
        
    #---------------------------------------------------------------------------
    def _create_augmented_X(s, X: pd.DataFrame) -> pd.DataFrame:
        """ Returns a new X DataFrame which has divergent features split
        into two. s.meta_feat and s.divergent_feats should be set prior
        to calling (which is usually done by fit). """
        if s._div_feats and s._meta_feat:
            non_div = list(set(X.columns) - set(s._div_feats))
            newX = X[non_div].copy()
            #new_feats = product(s.div_feats, ['_m','_f'])
            for feat in s._div_feats:
                assert(feat in X.columns)
                male_feat = feat + '_m'
                female_feat = feat + '_f'
                newX[male_feat] = X[feat]*X[s._meta_feat]
                newX[female_feat] = X[feat]*(X[s._meta_feat] == 0)
            return newX
        else:
            return X    
                  
#-------------------------------------------------------------------------------
def test():
    """ A simple test of the Meta class which also compares results to logistic
    regression. """
    df = pd.read_csv('test/hs_avg.csv', skipinitialspace=True)
    df.rename(columns={'truth':'y'}, inplace=True)
    X = df[['anger','contempt','disgust','engagement','fear', 'sadness',
            'surprise', 'is_male']]
    y = df['y']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5)
    
    clf = LogisticRegression(penalty='l1', C=1)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    acc = metrics.accuracy_score(y_pred,y_test)
    print('-----------------------------------')
    print('LR accuracy\t: %.4f' % acc)

    clf = Meta(penalty='l1', C=1)
    clf.fit(X_train, y_train, div_feats=['engagement', 'anger'], 
            meta_feat='is_male')
    y_pred = clf.predict(X_test)
    acc = metrics.accuracy_score(y_pred,y_test)
    print('-----------------------------------')
    print('Meta accuracy\t: %.4f' % acc)
    print('-----------------------------------')
    print('meta feature: ', clf._meta_feat)
    print('div features: ', clf._div_feats)
    
    # test score()
    print('score: %.4f' % clf.score(X_test, y_test))
    
    #-------------------------------------------------------------------------------
if __name__ == '__main__':
    print('starting Meta test')
    test()
    print('Meta test complete')
