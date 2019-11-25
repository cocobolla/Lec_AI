import pandas as pd
import numpy as np
import os
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score


def main():
    root_path = r'C:\Users\USER\workspace\Lec\Lec_AI\HW2_data'
    file_name = 'default.csv'
    file_path = os.path.join(root_path, file_name)

    df = pd.read_csv(file_path)

    # 3 - (1)
    df['default'] = np.where(df['default'] == "Yes", 1, 0)
    df['student'] = np.where(df['student'] == "Yes", 1, 0)

    scaler = StandardScaler()
    scaler.fit(df[['balance']])
    df["balance"] = scaler.transform(df[['balance']])
    scaler.fit(df[['income']])
    df["income"] = scaler.transform(df[['income']])

    # 3 - (2)
    x = df.loc[:, 'student':'income']
    y = df['default']

    lr = LogisticRegression()
    result = lr.fit(x, y)

    # 3 - (3)
    intercept, coefficients = result.intercept_, result.coef_
    print('Coef:', coefficients)
    print('Intercept', intercept)

    ah = intercept[0].round(3)  # alpha_hat
    bh1 = coefficients[0][0].round(3)  # beta_hat1
    bh2 = coefficients[0][1].round(3)  # beta_hat2
    bh3 = coefficients[0][2].round(3)  # beta_hat3

    print('default_prob = {} + {}*student + {}*balance + {}*income'.format(ah, bh1, bh2, bh3))

    # 3 - (4)
    lr.fit(df.iloc[:, 1:], df["default"])
    coef = np.array(lr.coef_)
    coef = coef.reshape(3, )
    print(coef)

    lr_preds_proba = lr.predict_proba(df.iloc[:, 1:])
    print(lr_preds_proba[:, 1])

    # 3 - (5)
    score = cross_val_score(lr, df.iloc[:, 1:], df["default"], scoring='accuracy', cv=3)
    print(np.mean(score))


if __name__ == '__main__':
    main()
