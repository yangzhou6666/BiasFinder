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

df = pd.read_csv("../asset/sst/misc/train_with_sentiment.csv", header = 0, sep=",")
can_mutate = []
with open("../asset/sst/train_0.6_0.4_split.csv", 'w') as outfile:
        writer = csv.writer(outfile, delimiter="\t")
        for index, row in df.iterrows():
            sentiment = row["sentiment"]
            if sentiment >= 0.6:
                label = 1
            elif sentiment <= 0.4:
                label = 0
            elif sentiment > 0.4 and sentiment < 0.6:
                continue
            text = row["sentence"]
            mg = MutantGeneration(text)
            if len(mg.getMutants()) > 0:
                #has mutants
                can_mutate.append([text, sentiment])
                continue
            else:
                #no mutants
                writer.writerow([label, text])

with open("../asset/sst/misc/test+mutable_train_with_sentiment.csv", 'a', newline="") as outfile:
        writer = csv.writer(outfile, delimiter=",")
        for instance in can_mutate:
            writer.writerow([instance[0], instance[1]])

python fine-tune.py \
  --task_name binary \
  --do_lower_case \
  --fine_tune_data_dir ./../../asset/sst/ \
  --vocab_file ./../../models/uncased_L-12_H-768_A-12/vocab.txt \
  --bert_config_file ./../../models/uncased_L-12_H-768_A-12/bert_config.json  \
  --train_batch_size 32 \
  --learning_rate 2e-5 \
  --num_train_epochs 20 \
  --seed 42 \
  --layers 11 10 \
  --trunc_medium -1 \
  --init_checkpoint ./../../models/fine-tuning/pytorch_bert_base_model.bin \
  --save_model_dir ./../../models/fine-tuning/pytorch_sst_fine_tuned_20_epoch_updated_training_set/









