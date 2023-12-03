import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import cross_val_score

df = pd.read_csv('E:\capstone project\insurance.csv')
df.head()

# Data description
df.describe()

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.ensemble import GradientBoostingRegressorsr
from sklearn import set_config

set_config(display='diagram')

df.head(2)

transformer = ColumnTransformer(transformers=[
    ('tnf1', OneHotEncoder(sparse=False, handle_unknown='ignore'), [1, 5]),
    ('tnf2', OrdinalEncoder(categories=[['no', 'yes']]), [4]),
    ('tnf3', StandardScaler(), [0, 2, 3])
], remainder='passthrough')

x_train, x_test, y_train, y_test = train_test_split(df.drop(columns=['charges']),
                                                    df['charges'], test_size=0.2, random_state=42)

x_train.shape
x_test.shape


gb_model = Pipeline(steps=[('transformer', transformer),
                           ('model', GradientBoostingRegressor())
                          ]) 

gb_model.fit(x_train, y_train)
y_pred = gb_model.predict(x_test)
y_pred

from sklearn.metrics import mean_squared_error, r2_score

gradient_boosting_mse = mean_squared_error(y_test, y_pred)
gradient_boosting_rmse = mean_squared_error(y_test, y_pred, squared=False)
gradient_boosting_r2_score = r2_score(y_test, y_pred)

print("The Mean Squared Error using Gradient Boosting Regressor: {}".format(gradient_boosting_mse))
print("The Root Mean Squared Error using Gradient Boosting Regressor: {}".format(gradient_boosting_rmse))
print("The R2 score using Gradient Boosting Regressor: {}".format(gradient_boosting_r2_score))

import pickle

pickle.dump(gb_model, open('gb_model.pkl', 'wb'))
