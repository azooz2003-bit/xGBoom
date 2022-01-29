from operator import index
from pathlib import Path
import sklearn
import numpy as np
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn import linear_model
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score
import pandas as pd
import os, sys 
import matplotlib.pyplot as plt
import datetime
import time

 
# Features we like: X, Y, result, situation, shot type, last action, minute(correlation with fatigue of players on pitch)

#proj_dir = Path("Desktop/xGBoost")
data_csv = "shots_dataset.csv"
complete_df = pd.read_csv(data_csv)

useful_stats = complete_df[["X","Y","result","situation","shotType"]]

# Result:
# 1 -> Goal
# 2 -> SavedShot
# 3 -> MissedShots
# 4 -> BlockedShot
# 5 -> ShotOnPost
# 6 -> OwnGoal
# 7 -> Own1

useful_stats.loc[:, "result"] = useful_stats['result'].replace({'Goal': '1'}, regex=True)
useful_stats.loc[:, "result"] = useful_stats['result'].replace({'SavedShot': '2'}, regex=True)
useful_stats.loc[:, "result"] = useful_stats['result'].replace({'MissedShots': '3'}, regex=True)
useful_stats.loc[:, "result"] = useful_stats['result'].replace({'BlockedShot': '4'}, regex=True)
useful_stats.loc[:, "result"] = useful_stats['result'].replace({'ShotOnPost': '5'}, regex=True)
useful_stats.loc[:, "result"] = useful_stats['result'].replace({'OwnGoal': '6'}, regex=True)
useful_stats.loc[:, "result"] = useful_stats['result'].replace({'Own1': '7'}, regex=True)

train, test = train_test_split(useful_stats, random_state=42, test_size=0.2, shuffle=True)

train_path = Path("", "train.csv")

test_path = Path("", "test.csv")

train.to_csv(train_path, index = False) # index number not displayed with values
test.to_csv(test_path, index = False) # index number not displayed with values

train_data = pd.read_csv("train.csv")
test_data = pd.read_csv("test.csv")

#regr.fit(X,Y) # we need to convert strings to numbers that represent them !!!!!

X_train = train_data[["X","Y","situation","shotType"]]
Y_train = train_data[["result"]]

X_test = test_data[["X","Y","situation","shotType"]]
Y_test = test_data[["result"]]

regr = linear_model.LinearRegression()

# To convert string data to graphable data
#ohe = OneHotEncoder(sparse=False)
#ohe.fit_transform(complete_df[["situation","shotType"]])
#print(ohe.categories_)

column_trans_trainX = make_column_transformer((OneHotEncoder(), ["situation","shotType"]), remainder="passthrough")
column_trans_trainX.fit_transform(X_train)
print(column_trans_trainX._columns)

pipe = make_pipeline(column_trans_trainX, regr)

print(cross_val_score(pipe, X_test, Y_test, cv=5).mean())

pipe.fit(X_train,Y_train)

Y_pred = pipe.predict(X_test) #pipeline is doing the dummy encoding of the new data, i.e. the switching of strings with nums

#print(accuracy_score(y_true= Y_test, y_pred= Y_pred)) <- doesn't work on lin reg


#print(useful_stats.head)

# Random graphs
plt.scatter(complete_df["X"], complete_df["result"])
plt.title("X Position vs Result of Shot")
plt.xlabel('X Position', fontsize = 14)
plt.ylabel("Result of Shot", fontsize = 14)
plt.grid(True) # to show grid lines
#plt.show()





#print(useful_stats.corr())


            
        




