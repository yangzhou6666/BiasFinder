import pandas as pd
from GenderDecider import GenderDecider

train = pd.read_csv("../../asset/imdb/train.csv", names=["label", "sentence"], sep="\t")
gender_distribution_train = {"male": 0, "female": 0, "cannot decide": 0, "no gender": 0}

for index, row in train.iterrows():
    text = row["sentence"]
    obj = GenderDecider(text)
    gender = obj.getGender()
    gender_distribution_train[gender] += 1

print("Gender Distribution Train" , gender_distribution_train)   

test = pd.read_csv("../../asset/imdb/test.csv", names=["label", "sentence"], sep="\t")
gender_distribution_test = {"male": 0, "female": 0, "cannot decide": 0, "no gender": 0}

for index, row in test.iterrows():
    text = row["sentence"]
    obj = GenderDecider(text)
    gender = obj.getGender()
    gender_distribution_test[gender] += 1

print("Gender Distribution Test" , gender_distribution_test) 
