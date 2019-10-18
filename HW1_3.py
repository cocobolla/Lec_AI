import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error

houses = fetch_california_housing()

X = pd.DataFrame(houses.data, columns=houses.feature_names)
y = pd.DataFrame(houses.target, columns=["target"])

scaler = StandardScaler()
scaler.fit(X)

# y_target = np.exp(houses['target'])
y_target = houses.target
X_data = scaler.transform(X)

# Question.4 - (1)
X_train, X_test, y_train, y_test = train_test_split(X_data, y_target, test_size=0.4, random_state=123)

# Question.4 - (2)
alpha_list = [0, 1, 10, 30, 50, 100]
for al in alpha_list:
    ridge = Ridge(alpha=al)
    neg_mse_scores = cross_val_score(ridge, X_train, y_train,
                                     scoring='neg_mean_squared_error', cv=5)
    avg_rmse = np.mean(np.sqrt(-1*neg_mse_scores))
    print('alpha={} -> RMSE={}'.format(al, np.around(avg_rmse, decimals=7)))

opt_al = 0

# Question.4 - (3)
ridge = Ridge(alpha=opt_al)
ridge.fit(X_train, y_train)
print("Parameter Estimation:")
print(ridge.coef_)
print("Intercept Estimation: {:.4f}".format(ridge.intercept_))

y_preds = ridge.predict(X_test)
sse = np.sum((y_test - y_preds)**2)
sst = np.sum((y_test - y_test.mean())**2)
r_sqaure = 1 - sse/sst

print("R Square: {:.4f}".format(r_sqaure))




fig, axs = plt.subplots(figsize=(18,6), nrows=1, ncols=len(alpha_list))
coeff_df = pd.DataFrame()
for pos, al in enumerate(alpha_list):
    ridge = Ridge(alpha=al)
    ridge.fit ( X_data, y_target)
    coeff = pd.Series(data=ridge.coef_, index=houses['feature_names'])
    colname = 'alpha:'+str(al)
    coeff_df[colname]=coeff
    coeff = coeff.sort_values(ascending=False)
    axs[pos].set_title(colname)
    axs[pos].set_xlim(-3,6)
    sns.barplot(x=coeff.values, y=coeff.index, ax=axs[pos])

plt.show()
