'''
This file tries to answer RQ1: Does bias hurt accuracy?
'''

from sentiment_analysis import SentimentAnalysis
from bias_rv.BiasRV import biasRV
import pandas as pd
from tqdm import tqdm

model_checkpoint='./../models/fine-tuning/pytorch_imdb_fine_tuned/epoch5.pt'
bert_config_file='./../models/uncased_L-12_H-768_A-12/bert_config.json'
vocab_file='./../models/uncased_L-12_H-768_A-12/vocab.txt'

### initialize an SA system
sa_system = SentimentAnalysis(model_checkpoint=model_checkpoint,
                            bert_config_file=bert_config_file,
                            vocab_file=vocab_file)

df = pd.read_csv("../asset/imdb/test.csv", names=["label", "sentence"], sep="\t")

path_to_result = '../result/original_dataset.txt'
# Index, true label, predicted label

analyze_original_performacne = False

rv = biasRV(sa_system.predict,X=4,alpha=0.1)

with open(path_to_result, 'w') as f:
    for index, row in  tqdm(df.iterrows(), desc="Evaluate"):
        label = row["label"]
        text = row["sentence"]
        result, is_bias = rv.verify(text)
        to_write = str(index) + ',' + str(label) + ',' + str(result) + ',' + str(is_bias) + '\n'
        f.write(to_write)




if analyze_original_performacne:
    with open(path_to_result, 'r') as f:
        lines = f.readlines()
        total_count = len(lines)
        correct_count = 0
        for line in lines:
            true_label = line.split(',')[1]
            pred_label = line.split(',')[2].strip()
            if true_label == pred_label:
                correct_count += 1
        print("Correct Predictions: ", correct_count)
        print("Total Predictions: ", total_count)
        print("Accuracy: ", 1.0 * correct_count / total_count)