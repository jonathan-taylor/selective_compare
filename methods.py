from __future__ import division, print_function

import time
import numpy as np
import statsmodels.api as sm
import pandas as pd
import rpy2.robjects as rpy

rpy.r('''
library(glmnet)
library(knockoff)
''')

# this code below is available at https://github.com/jonathan-taylor/selective-inference
from selection.algorithms.lasso import lasso
from selection.algorithms.sqrt_lasso import choose_lambda

def sqrt_lasso(X, Y, kappa, q=0.2):

    toc = time.time()
    lam = choose_lambda(X)
    L = lasso.sqrt_lasso(X, Y, kappa * lam)
    L.fit()
    S = L.summary('onesided')
    tic = time.time()

    selected = sm.stats.multipletests(S['pval'], q, 'fdr_bh')[0]

    return {'method':[r'$\kappa=%0.2f' % kappa],
            'active':[S['variable']], 
            'active_signs':[L.active_signs], 
            'pval':[S['pval']], 
            'selected':[selected],
            'runtime':tic-toc}

def cv_glmnet(X, Y, lam=['1SE', 'minCV'], q=0.2):

    rpy.r.assign('X', X)
    rpy.r.assign('Y', Y)
    
    toc = time.time()

    rpy.r('''
    Y = as.matrix(Y)
    X = as.matrix(X)
    G_CV = cv.glmnet(X, Y, standardize=FALSE, intercept=FALSE)

    lamhat = G_CV$lambda.min
    fit = glmnet(X, Y, standardize=FALSE,intercept=FALSE)
    Yhat = predict(fit, X, s = lamhat)
    nz = sum(predict(fit, s = lamhat, type = "coef") != 0)
    sigma_hat = sqrt(sum((Y - Yhat)^2)/(length(Y) - nz - 1))
    ''')

    tic = time.time()

    result = {'method':[],
              'active':[],
              'pval':[],
              'active_signs':[], 
              'selected':[],
              'runtime':[tic-toc, tic-toc]}

    sigma_hat = float(rpy.r('sigma_hat'))
    lam_1SE = float(rpy.r('G_CV$lambda.1se'))
    lam_minCV = float(rpy.r('G_CV$lambda.min'))

    lam_choice = {'1SE':lam_1SE,
                  'minCV':lam_minCV}

    for i, l in enumerate(lam):
        toc = time.time()
        L = lasso.gaussian(X, Y, lam_choice[l], sigma=sigma_hat)           
        L.fit() 
        S = L.summary('onesided')
        tic = time.time()

        selected = sm.stats.multipletests(S['pval'], q, 'fdr_bh')[0]

        results['method'].append(l)
        results['active'].append(S['variable'])
        results['active_signs'].append(L.active_signs)
        results['pval'].append(S['pval'])
        results['selected'].append(selected)
        results['runtime'][i] += tic-toc

    return results

def knockoff(X, Y, method=['knockoff', 'knockoff+'], q=0.2):

    rpy.r.assign('X', X)
    rpy.r.assign('Y', Y)
    rpy.r.assign('q', q)
    
    results = {'method':[],
               'active':[],
               'active_signs':[],
               'pval':[],
               'selected':[],
               'runtime':[]}

    for m in method:
        toc = time.time()
        rpy.r('''
        Y = as.matrix(Y)
        Y = as.numeric(Y)
        X = as.matrix(X)
        KO = knockoff.filter(X=X, y=Y, fdr=q, threshold="%s")
        ''')

        selected_idx = rpy.r('KO$selected')
        selected = np.zeros(p, np.bool)
        if selected_idx is not None:
            selected_idx = np.array(selected_idx) - 1 # R vs. python indexing
            selected[selected_idx] = True

        results['method'].append(m)
        results['active'].append(None)
        results['active_signs'].append(None)
        results['pval'].append([])
        results['selected'].append(selected)
        results['runtime'][i] += tic-toc

    return results

