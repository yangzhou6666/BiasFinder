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

df = pd.read_csv("../asset/imdb/misc/train_original.csv", names = ["label", "sentence"], sep="\t")
can_mutate = []
with open("../asset/imdb/train.csv", 'w', encoding='utf8') as outfile:
        writer = csv.writer(outfile, delimiter="\t")
        for index, row in df.iterrows():
            label = row["label"]
            sentence = row["sentence"]
            mg = MutantGeneration(sentence)
            if len(mg.getMutants()) > 0:
                #has mutants
                can_mutate.append([label, sentence])
                continue
            else:
                #no mutants
                writer.writerow([label, sentence])

with open("../asset/imdb/test+mutable_train.csv", 'a', newline="", encoding='utf8') as outfile:
        writer = csv.writer(outfile, delimiter="\t")
        for instance in can_mutate:
            writer.writerow([instance[0], instance[1]])

python fine-tune.py \
  --task_name binary \
  --do_lower_case \
  --fine_tune_data_dir ./../../asset/imdb/ \
  --vocab_file ./../../models/uncased_L-12_H-768_A-12/vocab.txt \
  --bert_config_file ./../../models/uncased_L-12_H-768_A-12/bert_config.json  \
  --train_batch_size 32 \
  --learning_rate 2e-5 \
  --num_train_epochs 5 \
  --seed 42 \
  --layers 11 10 \
  --trunc_medium -1 \
  --init_checkpoint ./../../models/fine-tuning/pytorch_bert_base_model.bin \
  --save_model_dir ./../../models/fine-tuning/pytorch_imdb_fine_tuned_0.5_split_5_epoch_updated_training_set/









