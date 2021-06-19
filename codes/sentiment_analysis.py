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


if __name__ == '__main__':
    '''and test scripts'''
    import pandas as pd
    ### initialize an SA system
    model_checkpoint='./../models/fine-tuning/pytorch_imdb_fine_tuned/epoch5.pt'
    bert_config_file='./../models/uncased_L-12_H-768_A-12/bert_config.json'
    vocab_file='./../models/uncased_L-12_H-768_A-12/vocab.txt'

    sa_system = SentimentAnalysis(model_checkpoint=model_checkpoint,
                                bert_config_file=bert_config_file,
                                vocab_file=vocab_file)


    df = pd.read_csv("../asset/imdb/test.csv", names=["label", "sentence"], sep="\t")
    # original test set
    texts = []
    for index, row in df.iterrows():
        label = row["label"]
        text = row["sentence"]
        texts.append(text)
        # result = sa_system.get_confidence(text)


    mutant_dir = "../data/biasfinder/gender/each/" 
    # the folder that stores generated mutants.

    df = pd.read_csv("../asset/imdb/test.csv", names=["label", "sentence"], sep="\t")
    # original test set

    alpha = 0.05   # specify "tolerance to bias"
    path_to_result = '../result/result_0.001.txt'

    correct_cnt = 0

    cnt = 0
    repair_cnt = 0
    
    tb = PrettyTable()
    tb.field_names = ["index", "correct", "difference"]
    with open(path_to_result) as f:
        lines = f.readlines()
        for line in lines:
            index = int(line.split(',')[0])
            ground = int(line.split(',')[1])
            predict = int(line.split(',')[2])
            is_bias = line.split(',')[-1]

            if not is_bias.strip() == "False":
                text = texts[int(index)]
                result = sa_system.get_confidence(text)
                original_pos_confidence = result[1]

                # get all the mutants
                path_to_mutant = mutant_dir + str(index) + '.csv'


                mutants = []
                if os.path.exists(path_to_mutant):
                    # if there are generated mutants

                    df_mutant = pd.read_csv(path_to_mutant, names=["label", "sentence"], sep="\t")
                    for index_new, row_new in df_mutant.iterrows():
                        mutants.append(row_new["sentence"])


                    mutant_len = len(mutants)
                    # if mutant_len < 60:
                    #     continue


                    confidences = []
                    pos_conf = 0.0
                    neg_conf = 0.0

                    # male

                    for m in mutants[0:int(mutant_len/2)]:
                        result = sa_system.get_confidence(m)
                        neg_conf += result[0]
                        pos_conf += result[1]
                        confidences.append(result)
                    
                    male_pos_confidence = pos_conf / len(confidences)
                    
                    # female
                    confidences = []
                    pos_conf = 0.0
                    neg_conf = 0.0

                    for m in mutants[int(mutant_len/2):]:
                        result = sa_system.get_confidence(m)
                        neg_conf += result[0]
                        pos_conf += result[1]
                        confidences.append(result)

                    female_pos_confidence = pos_conf / len(confidences)

                    diff = round(male_pos_confidence - female_pos_confidence, 2)

                    if abs(diff) > 0.12:
                        alpha = 1
                        beta = 0.1
                        final_confi =  male_pos_confidence * alpha + female_pos_confidence * (1 - alpha)
                        new_result = 1 if final_confi >= 0.5 else 0
                        print("----------")
                        print(index)
                        print(new_result == ground)

                        print(original_pos_confidence)
                        print(final_confi)
                        if new_result == ground:
                            repair_cnt += 1


                        


                    tb.add_row([index, str(ground == predict), diff])


                    
                    if (male_pos_confidence - original_pos_confidence) > (female_pos_confidence - original_pos_confidence):
                        if ground == predict:
                            correct_cnt += 1
                    cnt += 1

    print(tb)
    print(correct_cnt)
    print("repair cnt: ", repair_cnt)

    print(cnt)
                    

                    
