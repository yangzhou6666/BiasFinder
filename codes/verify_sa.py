# -*- coding: utf-8 -*-  
import pandas as pd
from sentiment_analysis import SentimentAnalysis
import time
import sys
from utils import preprocessText
from MutantGeneration import MutantGeneration
from matplotlib import pyplot as plt 


model_checkpoint='./../models/fine-tuning/pytorch_imdb_fine_tuned/epoch5.pt'
bert_config_file='./../models/uncased_L-12_H-768_A-12/bert_config.json'
vocab_file='./../models/uncased_L-12_H-768_A-12/vocab.txt'

def analyze_mut_generate_time():
    df = pd.read_csv("../asset/imdb/test.csv", names=["label", "sentence"], sep="\t")
    length_and_time = []
    time_list = []
    length_list = []
    count = 0
    for index, row in df.iterrows():
        count += 1
        if count % 100 == 0:
            break
        label = row["label"]
        text = row["sentence"]
        text = preprocessText(text)
        start = time.time()
        mg = MutantGeneration(text)
        consumed_time = time.time() - start
        time_list.append(consumed_time)
        length_list.append(len(text))

    ### print information
    print(">>>>>> Analyzing mutation generation time")
    print("Max: ", max(time_list))
    print("Min: ", min(time_list))
    
    plt.scatter(length_list,time_list)
    plt.savefig('testblueline.jpg')






def analyze_overhead():
    ### initialize an SA system
    sa_system = SentimentAnalysis(model_checkpoint=model_checkpoint,
                                bert_config_file=bert_config_file,
                                vocab_file=vocab_file)
    df = pd.read_csv("../asset/imdb/test.csv", names=["label", "sentence"], sep="\t")
    count = 0
    original_time = []
    one_step_time = []
    two_step_time = []

    for index, row in df.iterrows():
        loop_start = time.time()
        count += 1
        if count % 1000 == 0:
            pass
        label = row["label"]
        text = row["sentence"]

        final_result, consumed_time, nb_mut = sa_system.one_step_biasRV(text, True)
        one_step_time.append(consumed_time)

        ### time for predict once
        start_time_predict_one = time.time()
        sa_system.predict(text)
        consumed_time_2 = time.time() - start_time_predict_one
        original_time.append(consumed_time_2)
        # print("Time for predict one", consumed_time_2)
        final_result, is_bias, sampled_male_mutants, sampled_female_mutants, male_mut_results, female_mut_results, consumed_time_3 = sa_system.predict_biasRV(text, True)
        two_step_time.append(consumed_time_3)
    
    assert len(original_time) == len(one_step_time)
    assert len(one_step_time) == len(two_step_time)

    print("How much overhead introduced by two-step verification strategy?")
    print("Total text analyzed: ", count)

    print("Original total time: ", sum(original_time))
    print("One Step total time: ", sum(one_step_time))
    print("Two Step total time: ", sum(two_step_time))

    print("Overhead caused by one step: ", sum(one_step_time) / sum(original_time))
    print("Overhead caused by two step: ", sum(two_step_time) / sum(original_time))

def analyze_total_overhead():
    ### initialize an SA system
    sa_system = SentimentAnalysis(model_checkpoint=model_checkpoint,
                                bert_config_file=bert_config_file,
                                vocab_file=vocab_file)
    df = pd.read_csv("../asset/imdb/test.csv", names=["label", "sentence"], sep="\t")
    count = 0
    original_time = []
    one_step_time = []
    two_step_time = []

    for index, row in df.iterrows():
        loop_start = time.time()
        count += 1
        if count % 100 == 0:
            pass
        label = row["label"]
        text = row["sentence"]
        
        start_time_one_step = time.time()
        final_result = sa_system.one_step_biasRV(text)
        consumed_time = time.time() - start_time_one_step
        one_step_time.append(consumed_time)

        ### time for predict once
        start_time_original = time.time()
        sa_system.predict(text)
        consumed_time_2 = time.time() - start_time_original
        original_time.append(consumed_time_2)

        start_time_two_step = time.time()
        final_result, is_bias= sa_system.predict_biasRV(text)
        consumed_time_3 = time.time() - start_time_two_step
        two_step_time.append(consumed_time_3)
    
    assert len(original_time) == len(one_step_time)
    assert len(one_step_time) == len(two_step_time)

    print("How much (total) overhead introduced by two-step verification strategy?")
    print("Total text analyzed: ", count)

    print("Original total time: ", sum(original_time))
    print("One Step total time: ", sum(one_step_time))
    print("Two Step total time: ", sum(two_step_time))

    print("Overhead caused by one step: ", sum(one_step_time) / sum(original_time))
    print("Overhead caused by two step: ", sum(two_step_time) / sum(original_time))

def analyze_one_step_perf():
    ### initialize an SA system
    sa_system = SentimentAnalysis(model_checkpoint=model_checkpoint,
                                bert_config_file=bert_config_file,
                                vocab_file=vocab_file)
    df = pd.read_csv("../asset/imdb/test.csv", names=["label", "sentence"], sep="\t")
    writer = open("one_step_bias.txt", 'w')
    
    count = 0
    bias_count = 0
    for index, row in df.iterrows():
        count += 1
        if count % 30 == 0:
            pass
        
        label = row["label"]
        text = row["sentence"]
        final_result, is_bias, total_result, consumed_time, nb_mutants, mutants = sa_system.one_step_biasRV(text, debugging=True)
        if not is_bias:
            bias_count += 1
            to_write = str(count) + '\t' + ">>>>>>>>>>" + '\n'
            to_write += "Original Text: " + '\n' + text + '\n'
            # write predicted results for original text
            to_write += "Predicted Results: " + str(total_result[0]) + '\n'
            to_write += "Ground truth: " + str(label) + '\n'
            print(to_write)

            # write mutants results
            middle = int((len(total_result)) / 2)
            if middle == 0: 
                print("Male mutant results: ", end='')
                print(total_result[ 1 : middle + 1])
                print("Fema mutant results: ", end='')
                print(total_result[middle + 1 : ])            

    
    print("Bias count: ", bias_count)
    print("Bias rate: ", 1.0 * bias_count / count)
        

if __name__=="__main__":
    analyze_one_step_perf()


