import sys
sys.path.append('./fine-tuning')
sys.path.append('./gender')

from modeling_single_layer import BertConfig, BertForSequenceClassification
import torch
import tokenization
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm, trange
import numpy as np
import random
import time
import torch.nn as nn
import math
import os
from prettytable import PrettyTable
from gender.MutantGeneration import MutantGeneration

seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)


def check_property_1(original_result, female_mut_results, male_mut_results, N):
    return sum(female_mut_results) == sum(male_mut_results) and sum(female_mut_results) == original_result * N

def repair_with_majority_rule(results):
    '''
    repair using majority rule
    Input: a group of prediction results
    Output: majority results
    '''
    return 1 if 1.0 * sum(results) / len(results) > 0.5 else 0

class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, label_id):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id

class SentimentAnalysis():
    '''Class for a sentiment analysis system'''

    def __init__(self, model_checkpoint, bert_config_file, vocab_file):
        '''
        Initialize a sentiment analysis engine
        Paramters
            model_checkpoint: path to the model checkpoint
            bert_config_file: The config json file corresponding to the pre-trained BERT model.
            vocab_file: The vocabulary file that the BERT model was trained on.
        '''
        ### set label list
        label_list = ["0", "1"]


        ### Initialize a bert model
        bert_config = BertConfig.from_json_file(bert_config_file)
        self._model = BertForSequenceClassification(bert_config, len([0,1]), [11, 10], pooling=None)
        self._model.bert.load_state_dict(torch.load(model_checkpoint, map_location='cpu'))
        self.device = torch.device("cuda")
        self._model.to(self.device)

        ### Initialize tokenizer
        self.tokenizer = tokenization.FullTokenizer(vocab_file=vocab_file, do_lower_case=True)

    def convert_text_to_feature(self, texts, max_seq_length=128, trunc_medium=-1):
        '''
        Convert text to features that BERT can take in
        '''
        features = []
        for text in texts:
            # 1: convert to unicode
            text = tokenization.convert_to_unicode(text)

            # 2: convert to features
            # eval_features = convert_examples_to_features(eval_examples, label_list, args.max_seq_length, tokenizer, trunc_medium=args.trunc_medium)
            tokens_a = self.tokenizer.tokenize(text)

            if len(tokens_a) > max_seq_length - 2:
                if trunc_medium == -2:
                    tokens_a = tokens_a[0:(max_seq_length - 2)]
                elif trunc_medium == -1:
                    tokens_a = tokens_a[-(max_seq_length - 2):]
                elif trunc_medium == 0:
                    tokens_a = tokens_a[:(max_seq_length - 2) // 2] + tokens_a[-((max_seq_length - 2) // 2):]
                elif trunc_medium > 0:
                    tokens_a = tokens_a[: trunc_medium] + tokens_a[(trunc_medium - max_seq_length + 2):]

            tokens = []
            segment_ids = []
            tokens.append("[CLS]")
            segment_ids.append(0)
            for token in tokens_a:
                tokens.append(token)
                segment_ids.append(0)
            tokens.append("[SEP]")
            segment_ids.append(0)

            input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
            input_mask = [1] * len(input_ids)

            # Zero-pad up to the sequence length.
            while len(input_ids) < max_seq_length:
                input_ids.append(0)
                input_mask.append(0)
                segment_ids.append(0)

            assert len(input_ids) == max_seq_length
            assert len(input_mask) == max_seq_length
            assert len(segment_ids) == max_seq_length

            label_id = -1
            ## since we don't know the ground truth of a text
            feature = InputFeatures(
                            input_ids=input_ids,
                            input_mask=input_mask,
                            segment_ids=segment_ids,
                            label_id=label_id)
            features.append(feature)

        all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
        all_label_ids = torch.tensor([f.label_id for f in features], dtype=torch.long)


        tensor_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
        data_loader = DataLoader(tensor_data, batch_size=1, shuffle=False)

        return data_loader
                
    
    def predict(self, text: str):
        '''
        predict the sentiment label of a list of texts
        Parameters
            text:         a piece of text
        '''

        data_loader = self.convert_text_to_feature([text])
        
        # covert text: str to features that bert can take in

        self._model.eval()

        device = self.device
        results = []

        for input_ids, input_mask, segment_ids, label_ids in data_loader:
            input_ids = input_ids.to(device)
            input_mask = input_mask.to(device)
            segment_ids = segment_ids.to(device)
            label_ids = label_ids.to(device)
            
            with torch.no_grad():
                logits = self._model(input_ids=input_ids, token_type_ids=segment_ids, attention_mask=input_mask, labels=None)
            logits = logits.detach().cpu().numpy()
            predicted_label = np.argmax(logits, axis=1)
            results.append(predicted_label[0])

        assert len(results) == 1

        final_result = results[0]
        
        return final_result

    def predict_batch(self, text: list):
        '''
        predict the sentiment label of a list of texts
        Parameters
            text:         a piece of text
        '''

        data_loader = self.convert_text_to_feature(text)
        # covert text: str to features that bert can take in

        self._model.eval()

        device = self.device
        results = []

        for input_ids, input_mask, segment_ids, label_ids in data_loader:
            input_ids = input_ids.to(device)
            input_mask = input_mask.to(device)
            segment_ids = segment_ids.to(device)
            label_ids = label_ids.to(device)
            
            with torch.no_grad():
                logits = self._model(input_ids=input_ids, token_type_ids=segment_ids, attention_mask=input_mask, labels=None)
            logits = logits.detach().cpu().numpy()
            predicted_label = np.argmax(logits, axis=1)
            results.append(predicted_label[0])

        assert len(results) == len(text)

        return results

    def get_confidence(self, text: str):
        '''
        predict the sentiment label of a list of texts
        Parameters
            text:         a piece of text
        return the confidence value, instead of final sentiment
        '''

        data_loader = self.convert_text_to_feature([text])
        # covert text: str to features that bert can take in

        self._model.eval()

        device = self.device
        results = []

        for input_ids, input_mask, segment_ids, label_ids in data_loader:
            input_ids = input_ids.to(device)
            input_mask = input_mask.to(device)
            segment_ids = segment_ids.to(device)
            label_ids = label_ids.to(device)
            
            with torch.no_grad():
                logits = self._model(input_ids=input_ids, token_type_ids=segment_ids, attention_mask=input_mask, labels=None)
            logits = logits.detach().cpu().numpy()
            a = logits[0][0]
            b = logits[0][1]

            exp_sum = math.exp(a) + math.exp(b)
            result = [math.exp(a) / exp_sum, math.exp(b) / exp_sum]


        
        return result

def minority_or_majority(mutants, predict, majority):
    
    results = []
    for m in mutants:
        results.append(sa_system.predict(m))

    freq_1 = 0
    freq_0 = 0
    for result in results:
        if result == 1:
            freq_1 += 1
        else:
            freq_0 += 1
    
    if freq_1 > freq_0:
        majority_result = 1
        minority_result = 0
    else:
        majority_result = 0
        minority_result = 1
    
    if majority == True:
        predict = majority_result
    else: 
        predict = minority_result
    
    return predict

def average_conf_strategy(mutants, predict):

    confidences = []
    pos_conf = 0.0

    for m in mutants:
        result = sa_system.get_confidence(m)
        pos_conf += result[1]
        confidences.append(result)
        
    pos_confidence = pos_conf / (len(confidences))
    
    if pos_confidence > 0.5:
        predict = 1
    
    return predict

def median_conf_strategy(mutants, predict):
  
    confidences = []
    pos_conf = 0.0
    # male

    for m in mutants:
        result = sa_system.get_confidence(m)
        confidences.append(result[1])
    confidences.sort(reverse = True)
    if not len(confidences) == 1:
        for conf in confidences[int(len(confidences)/4):int(len(confidences)*3/4)]:
            pos_conf += conf
        
        median_pos_confidence = pos_conf / (len(confidences)/2)
    else:
        for conf in confidences:
            pos_conf += conf
        
        median_pos_confidence = pos_conf / (len(confidences))
    
    if median_pos_confidence > 0.5:
        predict = 1
    
    return predict

def male_conf_strategy(mutants, predict):
  
    confidences = []
    pos_conf = 0.0
    # male

    for m in mutants[0:int(len(mutants)/2)]:
        result = sa_system.get_confidence(m)
        confidences.append(result[1])
    confs = sum(confidences)
    confidences.sort(reverse = True)
    if not len(confidences) == 1:
        for conf in confidences[0:int(len(confidences)/2)]:
            pos_conf += conf
        
        male_pos_confidence = pos_conf / (len(confidences)/2)
    else:
        for conf in confidences:
            pos_conf += conf
        
        male_pos_confidence = pos_conf / (len(confidences))
    
    if male_pos_confidence > 0.5:
        predict = 1
    
    return [predict, confs]


def female_conf_strategy(mutants, predict):
    
    # female
    confidences = []
    pos_conf = 0.0
    for m in mutants[int(len(mutants)/2):]:
        result = sa_system.get_confidence(m)
        # pos_conf += result[1]
        confidences.append(result[1])
    confidences.sort()
    confs = sum(confidences)

    if not len(confidences) == 1:
        for conf in confidences[0:int(len(confidences)/2)]:
            pos_conf += conf
        
        female_pos_confidence = pos_conf / (len(confidences)/2)
    else:
        for conf in confidences:
            pos_conf += conf
        
        female_pos_confidence = pos_conf / (len(confidences))
    
    if female_pos_confidence > 0.5:
        predict = 1
    
    return [predict, confs]
    

if __name__ == '__main__':
    '''and test scripts'''
    import pandas as pd
    ### initialize an SA system
    # model_checkpoint='./../models/fine-tuning/pytorch_imdb_fine_tuned/epoch5.pt'
    model_checkpoint='./../models/epoch20.pt'
    # model_checkpoint='./../models/epoch20.pt'
    bert_config_file='./../models/uncased_L-12_H-768_A-12/bert_config.json'
    vocab_file='./../models/uncased_L-12_H-768_A-12/vocab.txt'

    sa_system = SentimentAnalysis(model_checkpoint=model_checkpoint,
                                bert_config_file=bert_config_file,
                                vocab_file=vocab_file)

    df = pd.read_csv("../asset/new_sst_test.csv", header = 0, sep=",")
    # df = pd.read_csv("../asset/new_imdb_test.csv", names=["label", "sentence"], sep="\t")
    # original test set
    texts = []
    for index, row in df.iterrows():
        # label = row["label"]
        # texts.append(row["sentence"])
        text = row["sentence"]
        texts.append(text)

    mutant_dir = "../data/biasfinder/gender/sst/each3/each/" 
    # mutant_dir = "../data/biasfinder/gender/new/each1/"
    # the folder that stores generated mutants.

    # original test set

    alpha = 0.05   # specify "tolerance to bias"
    path_to_result = '../result/395_sst_result_0.001.txt'
    
    correct_cnt = 0
    average_conf_repair = 0
    male_bottom_repair = 0
    female_top_repair = 0
    tmplate_repair_cnt = 0
    majority_repair_cnt = 0
    minority_repair_cnt = 0
    cnt = 0
    median_conf_repair =0
    count = 0
    diff = []
    male_mutants = 0
    female_mutants = 0
    bigger_male_confident = 0
    all_male_confs = 0
    all_female_confs = 0
    tb = PrettyTable()
    tb.field_names = ["index", "correct", "difference"]
    with open(path_to_result) as f:
        lines = f.readlines()
        for line in lines:
            index = int(line.split(',')[0])
            ground = int(line.split(',')[1])
            predict = int(line.split(',')[2])
            is_bias = line.split(',')[-1]
            
            # if os.path.exists(path_to_mutant):
                # if there are generated mutants
                
                # df_mutant = pd.read_csv(path_to_mutant, names=["label", "sentence", "mutant"], sep="\t")
                # for index_new, row_new in df_mutant.iterrows():
                #     mutants.append(row_new["sentence"])
                #     template_content = row_new["mutant"]
                # # print(mutants)
                # if len(mutants)<=4:
                #     count += 1
            if not is_bias.strip() == "False":
                text = texts[int(index)]
                result = sa_system.get_confidence(text)
                original_pos_confidence = result[1]

                # get all the mutants
                path_to_mutant = mutant_dir + str(index) + '.csv'

                mutants = []
                if os.path.exists(path_to_mutant):
                    # if there are generated mutants

                    df_mutant = pd.read_csv(path_to_mutant, names=["label", "sentence", "mutant"], sep="\t")
                    for index_new, row_new in df_mutant.iterrows():
                        mutants.append(row_new["sentence"])
                        template_content = row_new["mutant"]
                    male_mutants += len(mutants)/2
                    female_mutants += len(mutants)/2
                    if ground == predict:
                        correct_cnt += 1
                    male_conf = male_conf_strategy(mutants, predict)
                    female_conf = female_conf_strategy(mutants, predict)
                    if ground == male_conf[0]:
                        male_bottom_repair += 1
                    if male_conf[1]>female_conf[1]:
                        bigger_male_confident += 1
                    if ground == female_conf[0]:
                        female_top_repair += 1

                    if ground == average_conf_strategy(mutants, predict):
                        average_conf_repair += 1
                    cnt += 1
                    all_male_confs += male_conf[1]
                    all_female_confs += female_conf[1]
                    
                    if ground == median_conf_strategy(mutants, predict):
                        median_conf_repair += 1
                    if ground == minority_or_majority(mutants, predict, True):
                        majority_repair_cnt += 1
                    
                    if ground == minority_or_majority(mutants, predict, False):
                        minority_repair_cnt += 1

    print(correct_cnt, correct_cnt/cnt)
    print(cnt)
    print("majority strategy:", majority_repair_cnt, majority_repair_cnt/cnt)
    print("minority strategy:", minority_repair_cnt, minority_repair_cnt/cnt)
    print("average conf strategy:", average_conf_repair, average_conf_repair/cnt)
    print("male bottom strategy:", male_bottom_repair, male_bottom_repair/cnt)
    print("female top strategy:", female_top_repair, female_top_repair/cnt)
    print("median top strategy:", median_conf_repair, median_conf_repair/cnt)
    if all_male_confs > all_female_confs:
        print("male confs are bigger", all_male_confs-all_female_confs, (all_male_confs-all_female_confs)/male_mutants)
    else:
        print("female confs are bigger", abs(all_male_confs-all_female_confs), abs(all_male_confs-all_female_confs)/female_mutants)
    print(bigger_male_confident)
                    

                    
