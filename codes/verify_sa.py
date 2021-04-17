# -*- coding: utf-8 -*-  
import pandas as pd
from sentiment_analysis import SentimentAnalysis
import time
import sys
from utils import preprocessText
from MutantGeneration import MutantGeneration
from matplotlib import pyplot as plt
import prettytable as pt
from textwrap import fill
import argparse

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
            pass
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

        final_result, is_bias, results, consumed_time, nb_mut, mutants = sa_system.one_step_biasRV(text, True)
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
    '''
    It logs all the cases that are labelled as "potentially biased".
    '''
    ### initialize an SA system
    sa_system = SentimentAnalysis(model_checkpoint=model_checkpoint,
                                bert_config_file=bert_config_file,
                                vocab_file=vocab_file)
    df = pd.read_csv("../asset/imdb/test.csv", names=["label", "sentence"], sep="\t")
    
    count = 0
    bias_count = 0
    for index, row in df.iterrows():
        count += 1
                
        label = row["label"]
        text = row["sentence"]
        final_result, is_bias, total_result, consumed_time, nb_mutants, mutants = sa_system.one_step_biasRV(text, debugging=True)
        if is_bias:
            original_table = pt.PrettyTable(["Type", "Label", "Original Content", "True Label"],align='l')
            bias_count += 1
            print('\n' + str(count) + '\t' + ">>>>>>>>>>" + '\n')
            original_table.add_row(["Origin", str(total_result[0]), fill(text,width=150), str(label)])
            print(original_table)

            # write mutants results
            middle = int((len(total_result)) / 2)
            if not middle == 0: 
                table = pt.PrettyTable(["Type", "Label", "Content"],align='l')

                print("Male mutant results: ", end='')
                male_results = total_result[ 1 : middle + 1]
                male_results.sort(reverse=True)
                print(male_results)
                print("Fema mutant results: ", end='')
                female_results = total_result[middle + 1 : ]
                female_results.sort(reverse=True)
                print(female_results)

                texts = [text] + mutants
                assert len(texts) == len(total_result)
                for index in range(len(total_result)):
                    if index == 0:
                        continue
                    if index <= middle:
                        table.add_row(['Male', str(total_result[index]), fill(texts[index],width=150)])
                    else:
                        table.add_row(['Female', str(total_result[index]), fill(texts[index],width=150)])

                print(table)    


    
    print("Bias count: ", bias_count)
    print("Bias rate: ", 1.0 * bias_count / count)
        

def analyze_two_step_per():
    '''
    Log all the cases labelled as "potentially biased" by 2-step verifcation strategy.
    '''
    ### initialize an SA system
    sa_system = SentimentAnalysis(model_checkpoint=model_checkpoint,
                                bert_config_file=bert_config_file,
                                vocab_file=vocab_file)
    df = pd.read_csv("../asset/imdb/test.csv", names=["label", "sentence"], sep="\t")
    count = 0
    bias_count = 0
    for index, row in df.iterrows():
        count += 1
        label = row["label"]
        text = row["sentence"]
        final_result, is_bias, is_satisfy_prop_1, original_result, sampled_male_mutants, sampled_female_mutants, male_mut_results, female_mut_results, consumed_time = sa_system.predict_biasRV(text, debugging=True)

        if is_bias:
            original_table = pt.PrettyTable(["Type", "Label", "Original Content", "True Label"],align='l')

            bias_count += 1
            print('\n' + str(count) + '\t' + ">>>>>>>>>>" + '\n')
            original_table.add_row(["Origin", str(original_result), fill(text,width=150), str(label)])
            print(original_table)

            # write mutants results
            if male_mut_results == None:
                total_result = [original_result]
            else:
                total_result = [original_result] + male_mut_results + female_mut_results
            
            middle = int((len(total_result)) / 2)
            if not middle == 0: 
                table = pt.PrettyTable(["Type", "Label", "Content"],align='l')

                print("Male mutant results: ", end='')
                male_results = total_result[ 1 : middle + 1]
                male_results.sort(reverse=True)
                print(male_results)
                print("Fema mutant results: ", end='')
                female_results = total_result[middle + 1 : ]
                female_results.sort(reverse=True)
                print(female_results)

                texts = [text] + sampled_male_mutants + sampled_female_mutants
                assert len(texts) == len(total_result)
                for index in range(len(total_result)):
                    if index == 0:
                        continue
                    if index <= middle:
                        table.add_row(['Male', str(total_result[index]), fill(texts[index],width=150)])
                    else:
                        table.add_row(['Female', str(total_result[index]), fill(texts[index],width=150)])

                print(table)                

    print("Bias count: ", bias_count)
    print("Bias rate: ", 1.0 * bias_count / count)


if __name__=="__main__":
    parser = argparse.ArgumentParser(description="Evaluate BiasRV")
    parser.add_argument('question', help='Select the question you want to evaluate')
    args = parser.parse_args()

    question = args.question
    if question == "analyze_mut_generate_time":
        analyze_mut_generate_time()
    if question == "analyze_overhead":
        analyze_overhead()
    if question == "analyze_total_overhead":
        analyze_total_overhead()
    if question == "analyze_one_step_perf":
        analyze_one_step_perf()
    if question == "analyze_two_step_per":
        analyze_two_step_per()





