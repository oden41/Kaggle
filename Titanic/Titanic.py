import pandas as pd
import numpy as np
import csv as csv
from sklearn.ensemble import RandomForestClassifier

# 訓練データの読み込み
train_df = pd.read_csv("Data/train.csv", header=0)

# Sexをダミー変数に変換(female = 0, Male = 1)
train_df["Gender"] = train_df["Sex"].map({"female": 0, "male": 1}).astype(int)
train_df.head(3)

# 年齢の欠損値は、年齢の平均値で補完する
median_age = train_df["Age"].dropna().median()
if len(train_df.Age[train_df.Age.isnull()]) > 0:
    train_df.loc[(train_df.Age.isnull()), "Age"] = median_age

# 学習に必要無い列を削除する
train_df = train_df.drop(["Name", "Ticket", "Sex", "SibSp", "Parch", "Fare", "Cabin", "Embarked", "PassengerId"],
                         axis=1)
train_df.head(3)

# テストデータの読み込み, Sexをダミー変数に変換
test_df = pd.read_csv("Data/test.csv", header=0)
test_df["Gender"] = test_df["Sex"].map({"female": 0, "male": 1}).astype(int)

# 年齢の欠損値は、年齢の平均値で補完する
median_age = test_df["Age"].dropna().median()
if len(test_df.Age[test_df.Age.isnull()]) > 0:
    test_df.loc[(test_df.Age.isnull()), "Age"] = median_age

# テストデータのPassengerId列を退避させ、テストデータの不要な列を削除する
ids = test_df["PassengerId"].values
test_df = test_df.drop(["Name", "Ticket", "Sex", "SibSp", "Parch", "Fare", "Cabin", "Embarked", "PassengerId"], axis=1)
test_df.head(3)

# ランダムフォレストで予測
train_data = train_df.values
test_data = test_df.values
model = RandomForestClassifier(n_estimators=100)
output = model.fit(train_data[0::, 1::], train_data[0::, 0]).predict(test_data).astype(int)

# 結果を"titanic_submit.csv"として書き出す
submit_file = open("Data/titanic_submit.csv", "w")
file_object = csv.writer(submit_file)
file_object.writerow(['PassengerId', 'Survived'])
file_object.writerows(zip(ids, output))
submit_file.close()
