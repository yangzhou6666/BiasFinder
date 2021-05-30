'''
This file tries to answer RQ1: Does bias hurt accuracy?
'''


from unicodedata import name
from sentiment_analysis import SentimentAnalysis
from bias_rv.BiasRV import biasRV
import pandas as pd
from tqdm import tqdm
import os
import random
import numpy as np

seed = 42
random.seed(seed)
np.random.seed(seed)


def check_bias(results, alpha):
    '''decide whether it's bias given prediction results of mutants'''
    is_bias = False
    length = len(results)

    if length == 1:
        # no mutants
        pass
    else:
        mid = int((length - 1) / 2)
        male_results = results[1:mid+1]
        female_results = results[mid+1:]

        assert(len(male_results) == len(female_results))

        pos_M = 1.0 * sum(male_results) / len(male_results)
        pos_F = 1.0 * sum(female_results) / len(female_results)
        ### verify property (2) |pos_M - pos_F| < alpha
        is_bias = False if abs(pos_M - pos_F) < alpha else True

    return is_bias

if __name__ == '__main__':

    model_checkpoint='./../models/fine-tuning/pytorch_imdb_fine_tuned/epoch5.pt'
    bert_config_file='./../models/uncased_L-12_H-768_A-12/bert_config.json'
    vocab_file='./../models/uncased_L-12_H-768_A-12/vocab.txt'

    ### initialize an SA system
    sa_system = SentimentAnalysis(model_checkpoint=model_checkpoint,
                                bert_config_file=bert_config_file,
                                vocab_file=vocab_file)
    
    data_dir = "../data/biasfinder/gender/each/" 

    df = pd.read_csv("../asset/imdb/test.csv", names=["label", "sentence"], sep="\t")

    path_to_result = '../result/original_dataset.txt'
    # Index, true label, predicted label

    analyze_original_performacne = False


    with open(path_to_result, 'w') as f:
        for index, row in tqdm(df.iterrows(), desc="Evaluate"):
            label = row["label"]
            text = row["sentence"] # original text
            path_to_mutant = data_dir + str(index) + '.csv'
            mutants = [text]
            if os.path.exists(path_to_mutant):
                # if there are generated mutants
                df_mutant = pd.read_csv(path_to_mutant, names=["label", "sentence"], sep="\t")
                for index_new, row_new in df_mutant.iterrows():
                    mutants.append(row_new["sentence"])
            results = []
            results = sa_system.predict_batch(mutants)

            is_bias = False
            is_bias = check_bias(results, alpha=0.1)

            to_write = str(index) + ',' + str(label) + ',' + str(results[0]) + ',' + str(is_bias) + '\n'
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