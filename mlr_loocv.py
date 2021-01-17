import numpy as np
from sklearn import linear_model
from sklearn.model_selection import KFold
import statsmodels.api as sm
import statsmodels

import pandas as pd
from os.path import join

import tkinter as tk
from tkinter import filedialog

root = tk.Tk()
root.withdraw()

dataset_path = filedialog.askdirectory()


def lassoCV(X, y, alphas, cv=None):
    '''
    alphas: array-like; the alpha values to be tested
    cv: int or None; if None, then LOOCV, if int then KFold with cv number of splits
    '''
    clf_lasso = linear_model.LassoCV(alphas=alphas, cv=cv).fit(X,y)
    return clf_lasso


def ridgeCV(X, y, alphas, cv=None):
    '''
    alphas: array-like; the alpha values to be tested
    cv: int or None; if None, then LOOCV, if int then KFold with cv number of splits
    '''
    clf_ridge = linear_model.RidgeCV(alphas=alphas, cv=cv).fit(X,y)
    return clf_ridge


def drop_nans(X, y):
    nan_indices = np.where(np.isnan(y))[0]
    X.drop(axis='index', index=nan_indices, inplace=True)
    y = y[~np.isnan(y)]
    return (X, y)


def get_x_and_y(radioisotope, psych_test='MMSE'):
    indep_df = pd.read_csv(join(dataset_path, 'AD', radioisotope, 'stats', 'output_'+psych_test.lower()+'.csv'))
    indep_df = pd.concat([indep_df, pd.read_csv(join(dataset_path, 'MCI', radioisotope, 'stats', 'output_'+psych_test.lower()+'.csv'))], ignore_index=True)
    indep_df = pd.concat([indep_df, pd.read_csv(join(dataset_path, 'CN', radioisotope, 'stats', 'output_'+psych_test.lower()+'.csv'))], ignore_index=True)
    indep_df.drop([indep_df.columns[i] for i in range(2)], axis=1, inplace=True)

    target_df = pd.read_csv(join(dataset_path, 'AD', radioisotope, 'stats', 'summary.csv'))
    target_df = pd.concat([target_df, pd.read_csv(join(dataset_path, 'MCI', radioisotope, 'stats', 'summary.csv'))], ignore_index=True)
    target_df = pd.concat([target_df, pd.read_csv(join(dataset_path, 'CN', radioisotope, 'stats', 'summary.csv'))], ignore_index=True)

    X = indep_df
    y = target_df[psych_test]
    X, y = drop_nans(X, y)
    X_num = X.to_numpy(copy=True)
    y_num = y.to_numpy(copy=True)

    return (X_num, y_num)


def loocv_loop(X, y, mode='ridge'):
    clf = linear_model.Ridge(alpha=2)   # choose one of lasso (L1) or ridge (L2), vary alpha, and check rmse
    if mode == 'lasso':
        clf = linear_model.Lasso(alpha=2)

    sum_sq_errors = 0
    N = len(X)

    pred = []
    actual = []
    for i in range(N):
        X_val, y_val = np.array([X[i]]), np.array([y[i]])
        actual.append(y_val)
        # print('X.shape: ', X.shape)
        X_train = np.delete(X, i, axis=0)
        y_train = np.delete(y, i)

        clf.fit(X_train, y_train)
        pred_y_val = clf.predict(X_val)
        pred.append(pred_y_val)

        sq_error = (pred_y_val - y_val)**2
        sum_sq_errors += sq_error

    rmse_val = np.sqrt(sum_sq_errors / N)
    stdev = np.std(pred)
    print('rmse_val: ', rmse_val)   # currently the dataset is random, so wouldn't make much sense
    print('stdev: ', stdev)


radioisotopes = ['AV45', 'PiB']

df = None

for radioisotope in radioisotopes:
    X, y = get_x_and_y(radioisotope)
    loocv_loop(X, y)
