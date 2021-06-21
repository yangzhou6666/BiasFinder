import sys
sys.path.append('./fine-tuning')
sys.path.append('./gender')
import pandas as pd
import numpy as np
import math
import spacy

from gender.utils import preprocessText
from gender.MutantGeneration import MutantGeneration
import csv

df = pd.read_csv("../asset/sst/misc/test+dev+mutable_train.csv", header = 0, sep=",")
with open("../asset/sst/misc/test+dev+mutable_train_without_neutral.csv", 'w') as outfile:
        writer = csv.writer(outfile, delimiter=",")
        writer.writerow(['sentence', 'sentiment'])
        for index, row in df.iterrows():
            sentiment = row["sentiment"]
            if sentiment >= 0.6 or sentiment <= 0.4:
                sentence = row["sentence"]
                writer.writerow([sentence, sentiment])