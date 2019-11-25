from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import learning_curve
from scipy import stats
from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import math
import csv
import os
from termcolor import colored

random_state = 123


def train_test_split_ts(X_data, y_data, test_size, random_state=None):
    # Split preserving time series
    train_size = int(len(X_data) * (1-test_size))
    X_train = X_data[:train_size]
    y_train = y_data[:train_size]
    X_test = X_data[train_size:]
    y_test = y_data[train_size:]

    return X_train, X_test, y_train, y_test


def draw_learning_curve(model, X_data, y_data, model_name=None):
    train_sizes = [1, 50, 100, 200, 300, 500]
    train_sizes, train_scores, validation_scores = learning_curve(estimator=model, X=X_data, y=y_data,
                                                                  train_sizes=train_sizes, cv=5,
                                                                 scoring='neg_mean_squared_error')

    train_scores_mean = -train_scores.mean(axis=1)
    validation_scores_mean = -validation_scores.mean(axis=1)
    plt.plot(train_sizes, train_scores_mean, label='Training Error')
    plt.plot(train_sizes, validation_scores_mean, label='Validation Error')
    plt.ylabel('MSE')
    plt.xlabel('Training set size')
    plt.title('Learning curves for {}'.format(model_name))
    plt.legend()
    # plt.xlim(0, len(X_data))
    plt.show()
    plt.close()


def ridge_regression(X_data, y_target, split_func=train_test_split):
    # Data Pre-processing
    col_list = X_data.columns

    scaler = StandardScaler()
    scaler.fit(X_data)
    X_data = scaler.transform(X_data)
    X_train, X_test, y_train, y_test = split_func(X_data, y_target, test_size=0.3, random_state=random_state)
    # X_train = X_train[(np.abs(stats.zscore(X_train)) < 3).all(axis=1)]
    not_outlier_index = (np.abs(stats.zscore(X_train)) < 3.5).all(axis=1)
    X_train = X_train[not_outlier_index]
    y_train = y_train[not_outlier_index]

    # Parameter Tuning
    alphas = np.arange(0, 150, 1)
    rmse_list = []
    for al in alphas:
        ridge = Ridge(alpha=al)
        neg_mse_score = cross_val_score(ridge, X_train, y_train, scoring='neg_mean_squared_error', cv=5)
        avg_rmse = np.mean(np.sqrt(-1 * neg_mse_score))
        rmse_list.append(avg_rmse)

        # print(colored('Hello', 'green'))
    plt.plot(alphas, rmse_list)
    plt.title("Ridge Alpha - RMSE")
    plt.xlabel("Alpha")
    plt.ylabel("RMSE")
    plt.show()
    plt.close()

    opt_al = 10
    # opt_al = 100
    ridge = Ridge(alpha=opt_al)
    draw_learning_curve(ridge, X_data, y_target, 'Ridge')

    ridge.fit(X_train, y_train)
    y_preds = ridge.predict(X_test)
    mse = np.sqrt(mean_squared_error(y_test, y_preds))
    print('-' * 20)
    print("Ridge(alpha={}) Result".format(opt_al))
    print("MSE: ", end='')
    print(colored(round(mse, 5), 'green'))

    # Visualize Feature Decay
    alpha_list = [0, 0.1, 1, 10, 20, 50, 100, 150, 200]
    fig, axs = plt.subplots(figsize=(18, 6), nrows=1, ncols=len(alpha_list))
    coeff_df = pd.DataFrame()
    for pos, al in enumerate(alpha_list):
        ridge = Ridge(alpha=al)
        ridge.fit(X_data, y_target)
        coeff = pd.Series(data=ridge.coef_, index=col_list)
        colname = 'alpha:' + str(al)
        coeff_df[colname] = coeff
        coeff = coeff.sort_values(ascending=False)
        axs[pos].set_title(colname)
        axs[pos].set_xlim(-0.05, 0.05)
        sns.barplot(x=coeff.values, y=coeff.index, ax=axs[pos])
    # plt.show()
    plt.close()

    plt.plot(range(len(y_preds)), y_test, label='Test', alpha=0.9)
    plt.plot(range(len(y_preds)), y_preds, label='Predict', alpha=0.7)
    plt.title('Ridge Prediction')
    plt.ylabel('Return')
    plt.legend()
    plt.show()
    plt.close()

    return ridge


def lasso_regression(X_data, y_target, split_func=train_test_split):
    # Data Pre-processing
    col_list = X_data.columns
    scaler = StandardScaler()
    scaler.fit(X_data)
    X_data = scaler.transform(X_data)
    X_train, X_test, y_train, y_test = split_func(X_data, y_target, test_size=0.3, random_state=random_state)

    not_outlier_index = (np.abs(stats.zscore(X_train)) < 3.5).all(axis=1)
    X_train = X_train[not_outlier_index]
    y_train = y_train[not_outlier_index]

    # Parameter Tuning
    rmse_list = []
    alphas = np.arange(0.0001, 0.02, 0.0005)
    for al in alphas:
        lasso = Lasso(alpha=al)
        neg_mse_score = cross_val_score(lasso, X_train, y_train, scoring='neg_mean_squared_error', cv=5)
        avg_rmse = np.mean(np.sqrt(-1 * neg_mse_score))
        rmse_list.append(avg_rmse)

    plt.plot(alphas, rmse_list)
    plt.title("LASSO Alpha - RMSE")
    plt.xlabel("Alpha")
    plt.ylabel("RMSE")
    plt.show()
    plt.close()

    opt_al = alphas[rmse_list.index(min(rmse_list))]

    lasso = Lasso(alpha=opt_al)
    # Draw Learning Curve
    draw_learning_curve(lasso, X_data, y_target, 'Lasso')
    lasso.fit(X_train, y_train)
    y_preds = lasso.predict(X_test)
    mse = np.sqrt(mean_squared_error(y_test, y_preds))
    print('-' * 20)
    print("LASSO(alpha={}) Result".format(opt_al))
    print("MSE: ", end='')
    print(colored(round(mse, 5), 'green'))

    # Visualize Feature Decay
    alpha_list = [0.0001, 0.001, 0.0021, 0.003, 0.005]
    fig, axs = plt.subplots(figsize=(18, 6), nrows=1, ncols=len(alpha_list))
    coeff_df = pd.DataFrame()
    for pos, al in enumerate(alpha_list):
        ridge = Ridge(alpha=al)
        ridge.fit(X_data, y_target)
        coeff = pd.Series(data=lasso.coef_, index=col_list)
        colname = 'alpha:' + str(al)
        coeff_df[colname] = coeff
        coeff = coeff.sort_values(ascending=False)
        axs[pos].set_title(colname)
        axs[pos].set_xlim(-0.01, 0.01)
        sns.barplot(x=coeff.values, y=coeff.index, ax=axs[pos])
    # plt.show()
    plt.close()

    plt.plot(range(len(y_preds)), y_test, label='Test', alpha=0.9)
    plt.plot(range(len(y_preds)), y_preds, label='Predict', alpha=0.7)
    plt.title('Lasso Prediction')
    plt.ylabel('Return')
    plt.legend()
    plt.show()
    plt.close()

    return lasso


def random_forest(X_data, y_target, split_func=train_test_split):
    col_list = X_data.columns
    X_train, X_test, y_train, y_test = split_func(X_data, y_target, test_size=0.3, random_state=random_state)

    # n_est = 150
    # n_est = 80
    n_est = 100
    max_depth = 5
    max_feat = 5
    min_samples_leaf = 10
    n_jobs = 20

    '''
    # Parameter Tuning
    X_t, X_v, y_t, y_v  = train_test_split(X_train, y_train, test_size=0.3, random_state=random_state)
    est_list = range(20, 200, 10)
    depth_list = range(1, 10, 1)
    leaf_list = range(1, 10, 1)
    feature_list = range(1, 5, 1)
    mse_list = []
    for l in leaf_list:
        rf_clf = RandomForestRegressor(n_estimators=n_est, max_depth=max_depth,
                                       # max_features=math.sqrt(len(col_list)),
                                       min_samples_leaf=l, n_jobs=n_jobs)
        rf_clf.fit(X_t, y_t)
        y_val = rf_clf.predict(X_v)
        mse = np.sqrt(mean_squared_error(y_v, y_val))
        mse_list.append(mse)

    plt.plot(leaf_list, mse_list)
    plt.title("Random Forest # Estimators - RMSE")
    plt.xlabel("# Estimators")
    plt.ylabel("RMSE")
    plt.show()
    plt.close()
    '''

    rf_clf = RandomForestRegressor(n_estimators=n_est, max_depth=max_depth,
                                   max_features=max_feat, min_samples_leaf=min_samples_leaf,
                                   random_state=random_state, n_jobs=n_jobs)

    # Draw Learning Curve
    draw_learning_curve(rf_clf, X_train, y_train, 'Random Forest')
    rf_clf.fit(X_train, y_train)

    y_preds = rf_clf.predict(X_test)
    mse = np.sqrt(mean_squared_error(y_test, y_preds))

    print('-' * 20)
    print("Random Forest Result")
    print("MSE: ", end='')
    print(colored(round(mse, 5), 'green'))

    # Feature Importance
    imp_values = pd.Series(rf_clf.feature_importances_, index=X_train.columns)
    imp_values = imp_values.sort_values(ascending=False)
    plt.figure(figsize=(8, 6))
    plt.title('Feature Importance')
    sns.barplot(x=imp_values, y=imp_values.index)
    # plt.show()
    plt.close()

    plt.plot(range(len(y_preds)), y_test, label='Test', alpha=0.9)
    plt.plot(range(len(y_preds)), y_preds, label='Predict', alpha=0.7)
    plt.title('Random Forest Prediction')
    plt.ylabel('Return')
    plt.legend()
    plt.show()
    plt.close()
    return rf_clf


def data_reader():
    # Load Data
    root_path = r'C:\Users\USER\workspace\Lec\Lec_AI\mid_project'
    train_file = 'train.csv'
    test_file = 'test.csv'
    train_path = os.path.join(root_path, train_file)
    test_path = os.path.join(root_path, test_file)
    df_train = pd.read_csv(train_path)
    df_test = pd.read_csv(test_path)

    return df_train, df_test


def main():
    # Load Data
    df_train, df_test = data_reader()
    # Data Analysis
    df_train.info()
    df_test.info()

    df_train.describe()
    df_test.describe()
    # df_train.drop(df_train.columns[df_train.columns.str.contains('unnamed', case=False)], axis=1, inplace=True)
    df_train = df_train.loc[:, ~df_train.columns.str.contains('^Unnamed')]
    df_test_m = df_test.loc[:, ~df_test.columns.str.contains('^Unnamed')]

    # Visualize, Feature Selection,
    not_outlier_index = (np.abs(stats.zscore(df_train)) < 3.5).all(axis=1)
    fig, axs = plt.subplots(figsize=(16, 8), ncols=3, nrows=2)
    col_list = df_train.columns
    col_list = col_list.drop(labels='TARGET')
    for i, feature in enumerate(col_list):
        row = int(i / 3)
        col = i % 3
        sns.regplot(x=feature, y='TARGET', data=df_train, ax=axs[row][col])
    for i, feature in enumerate(col_list):
        row = int(i / 3)
        col = i % 3
        sns.regplot(x=feature, y='TARGET', data=df_train[not_outlier_index],
                    ax=axs[row][col], scatter_kws={'alpha': 0.5})
    plt.show()
    plt.close()

    X_data = df_train.loc[:, df_train.columns != 'TARGET']
    y_target = df_train['TARGET']

    ridge = ridge_regression(X_data, y_target)
    lasso = lasso_regression(X_data, y_target)
    rf = random_forest(X_data, y_target)

    # ridge = ridge_regression(X_data, y_target, train_test_split_ts)
    # lasso = lasso_regression(X_data, y_target, train_test_split_ts)
    # rf = random_forest(X_data, y_target, train_test_split_ts)

    # Test set save with Random Forest
    test_preds = rf.predict(df_test_m)
    df_test['Ypred'] = test_preds
    df_test.to_csv('test_result_박찬주.csv')

    # Test Result with Random Forest
    plt.plot(range(len(test_preds)-1), df_test['lag.1'][1:], label='Test', alpha=0.9)
    plt.plot(range(len(test_preds)-1), test_preds[:-1], label='Predict', alpha=0.7)
    plt.title('Random Forest Prediction')
    plt.ylabel('Return')
    plt.legend()
    plt.show()
    plt.close()

    # Result
    ridge_mse = np.sqrt(mean_squared_error(df_test['lag.1'][1:], ridge.predict(df_test_m)[:-1]))
    lasso_mse = np.sqrt(mean_squared_error(df_test['lag.1'][1:], lasso.predict(df_test_m)[:-1]))
    rf_mse = np.sqrt(mean_squared_error(df_test['lag.1'][1:], test_preds[:-1]))

    print()
    print('-' * 20)
    print("Ridge Test MSE: ", end='')
    print(colored(round(ridge_mse, 5), 'green'))
    print("Lasso Test MSE: ", end='')
    print(colored(round(lasso_mse, 5), 'green'))
    print("RandFo Test MSE: ", end='')
    print(colored(round(rf_mse, 5), 'green'))


if __name__ == '__main__':
    main()
