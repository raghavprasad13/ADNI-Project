import numpy as np
from matplotlib import pyplot as plt
from sklearn import linear_model
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score
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


def loocv_loop(X, y, mode='ridge', alpha=2):
    clf = linear_model.Ridge(alpha=alpha)   # choose one of lasso (L1) or ridge (L2), vary alpha, and check rmse
    if mode == 'lasso':
        clf = linear_model.Lasso(alpha=alpha)

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

    val_rmse = np.sqrt(sum_sq_errors / N)
    val_r2 = r2_score(actual, pred)
    val_stdev = np.std(pred)
    print('val rmse: ', val_rmse)   
    print('val r2: ', val_r2)  
    print('val stdev: ', val_stdev)

    clf.fit(X, y)
    pred_y_train = clf.predict(X)
    sq_errors = (pred_y_train - y)**2
    train_rmse = np.sqrt(np.sum(sq_errors) / N)
    train_r2 = r2_score(y, pred_y_train)
    train_stdev = np.std(pred_y_train)
    print('train rmse: ', train_rmse)   
    print('train r2: ', train_r2) 
    print('train stdev: ', train_stdev)

    return train_rmse, val_rmse, train_r2, val_r2, train_stdev, val_stdev

radioisotopes = ['AV45', 'PiB']

df = None
alpha_range = np.arange(1,30,1)


for mode in ['ridge', 'lasso']:
	print("Mode - ", mode)
	TRAIN_av45_rmse_list = []
	TRAIN_pib_rmse_list = []
	TRAIN_av45_r2_list = []
	TRAIN_pib_r2_list = []
	TRAIN_av45_stdev_list = []
	TRAIN_pib_stdev_list = []
	VAL_av45_rmse_list = []
	VAL_pib_rmse_list = []
	VAL_av45_r2_list = []
	VAL_pib_r2_list = []
	VAL_av45_stdev_list = []
	VAL_pib_stdev_list = []
	for radioisotope in radioisotopes:
		print("Radioisotope:", radioisotope)
		X, y = get_x_and_y(radioisotope)
		for alpha in alpha_range:
			print("alpha: ", alpha)
			train_rmse, val_rmse, train_r2, val_r2, train_stdev, val_stdev = loocv_loop(X, y, 'ridge', alpha)
			if radioisotope == 'AV45':
				TRAIN_av45_rmse_list.append(train_rmse)
				TRAIN_av45_r2_list.append(train_r2)
				TRAIN_av45_stdev_list.append(train_stdev)
				VAL_av45_rmse_list.append(val_rmse)
				VAL_av45_r2_list.append(val_r2)
				VAL_av45_stdev_list.append(val_stdev)
			elif radioisotope == 'PiB':
				TRAIN_pib_rmse_list.append(train_rmse)
				TRAIN_pib_r2_list.append(train_r2)
				TRAIN_pib_stdev_list.append(train_stdev)
				VAL_pib_rmse_list.append(val_rmse)
				VAL_pib_r2_list.append(val_r2)
				VAL_pib_stdev_list.append(val_stdev)
	plt.figure()
	plt.plot(alpha_range, TRAIN_av45_rmse_list, 'blue')
	plt.plot(alpha_range, VAL_av45_rmse_list, 'orange')
	plt.xlabel('alpha (' + mode + ')' )
	plt.ylabel('Prediction RMSE (TRAIN set and LOOCV)')
	plt.title('AV45 - Train and Val RMSE; Mode: '+ mode)
	plt.legend(['TRAIN', 'VAL'])
	plt.show()

	plt.figure()
	plt.plot(alpha_range, TRAIN_av45_r2_list, 'blue')
	plt.plot(alpha_range, VAL_av45_r2_list, 'orange')
	plt.xlabel('alpha (' + mode + ')' )
	plt.ylabel('Prediction R2 score (TRAIN set and LOOCV)')
	plt.title('AV45 - Train and Val R2 score; Mode: '+ mode)
	plt.legend(['TRAIN', 'VAL'])
	plt.show()


	plt.figure()
	plt.plot(alpha_range, TRAIN_av45_stdev_list, 'blue')
	plt.plot(alpha_range, VAL_av45_stdev_list, 'orange')
	plt.xlabel('alpha (' + mode + ')' )
	plt.ylabel('Prediction std deviation (TRAIN set and LOOCV)')
	plt.title('AV45 - Train and Val prediction std deviation; Mode: '+ mode)
	plt.legend(['TRAIN', 'VAL'])
	plt.show()

	plt.figure()
	plt.plot(alpha_range, TRAIN_pib_rmse_list, 'blue')
	plt.plot(alpha_range, VAL_pib_rmse_list, 'orange')
	plt.xlabel('alpha (' + mode + ')' )
	plt.ylabel('Prediction RMSE (TRAIN set and LOOCV)')
	plt.title('PiB - Train and Val RMSE; Mode: '+ mode)
	plt.legend(['TRAIN', 'VAL'])
	plt.show()

	plt.figure()
	plt.plot(alpha_range, TRAIN_pib_r2_list, 'blue')
	plt.plot(alpha_range, VAL_pib_r2_list, 'orange')
	plt.xlabel('alpha (' + mode + ')' )
	plt.ylabel('Prediction R2 score (TRAIN set and LOOCV)')
	plt.title('PIB - Train and Val R2 score; Mode: '+ mode)
	plt.legend(['TRAIN', 'VAL'])
	plt.show()

	plt.figure()
	plt.plot(alpha_range, TRAIN_pib_stdev_list, 'blue')
	plt.plot(alpha_range, VAL_pib_stdev_list, 'orange')
	plt.xlabel('alpha (' + mode + ')' )
	plt.ylabel('Prediction std deviation (TRAIN set and LOOCV)')
	plt.title('PiB - Train and Val prediction std deviation; Mode: '+ mode)	
	plt.legend(['TRAIN', 'VAL'])
	plt.show()