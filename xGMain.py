from operator import index
from pathlib import Path
import sklearn
from sklearn.model_selection import train_test_split
import pandas as pd
import os, sys 
import matplotlib.pyplot as plt
import datetime
import time

 
# Features we like: X, Y, result, situation, shot type, last action, minute(correlation with fatigue of players on pitch)

#proj_dir = Path("Desktop/xGBoost")
data_csv = "shots_dataset.csv"
complete_df = pd.read_csv(data_csv)

useful_stats = complete_df[["X","Y","result","situation","shotType","lastAction","minute"]]

print(useful_stats.head)

train, test = train_test_split(useful_stats, random_state=42, test_size=0.2, shuffle=True)

train_path = Path("", "train.csv")

test_path = Path("", "test.csv")

train.to_csv(train_path, index = False) # index number not displayed with values

test.to_csv(test_path, index = False) # index number not displayed with values





            
        




