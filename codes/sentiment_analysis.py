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
from MutantGeneration import MutantGeneration
import time

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


    def one_step_biasRV(self, text: str, debugging=False):
        '''
        verify property (1) use all generated mutants.
        '''

        '''generate mutants use biasfinder'''
        mg = MutantGeneration(text)
        results = []

        start_time = time.time()
        if len(mg.getMutants()) == 0:
            ### if there is no mutants generated
            results.append(self.predict(text))

        if len(mg.getMutants()) > 0:
            ### if there are mutants generated
            male_mutants = mg.get_male_mutants()
            female_mutants = mg.get_female_mutants()
            assert len(male_mutants) == len(female_mutants)
            for each_mutant in [text] + male_mutants + female_mutants:
                results.append(self.predict(each_mutant))

        consumed_time = time.time() - start_time
        final_result = repair_with_majority_rule(results)

        if debugging:
            return final_result, consumed_time, len(mg.getMutants())
        
        return final_result
        


    def predict_biasRV(self, text: str, debugging=False):
        '''
        Use bias RV to verify at runtime
        Input: text: str
        Output: final_result, is_bias
        '''

        N = 4
        L = 16
        is_bias = False
        alpha = 0.1

        total_time = []

        '''generate mutants use biasfinder'''
        mg = MutantGeneration(text)

        start_time = time.time()

        if len(mg.getMutants()) == 0:
            ### if there is no mutants generated
            final_result = self.predict(text)

        is_satisfy_prop_1 = True
        is_satisfy_prop_2 = True
        male_mutants = None
        female_mutants = None
        male_mut_results = None
        female_mut_results = None
        sampled_male_mutants = None
        sampled_female_mutants = None

        if len(mg.getMutants()) > 0:
            ### if there are mutants generated
            male_mutants = mg.get_male_mutants()
            female_mutants = mg.get_female_mutants()
            assert len(male_mutants) == len(female_mutants)

            # if biasfinder only generates two mutants (one for each gender)
            if len(male_mutants) == 1:
                ### TODO: deal with 1 mutant situation
                ### for now, we just return the result of original texts
                final_result = self.predict(text)

            ### select N mutants from each gender
                ### what if mutants are not enough? e.g. only generate 1 but we need 4.
            if N + L > len(female_mutants):
                ### TODO: dealing with such situation
                final_result = self.predict(text)
            else:
                # random selection
                sampled_male_mutants = random.sample(male_mutants, N + L)
                sampled_female_mutants = random.sample(female_mutants, N + L)
                original_result = self.predict(text)

                ## processing male_mutants
                male_mut_results = []
                for each_text in sampled_male_mutants[0: N]:
                    male_mut_results.append(self.predict(each_text))
                
                ## processing female_mutants
                female_mut_results = []
                for each_text in sampled_female_mutants[0: N]:
                    female_mut_results.append(self.predict(each_text))

                ### verify property (1)
                is_satisfy_prop_1 = check_property_1(original_result, female_mut_results, male_mut_results, N)
                if is_satisfy_prop_1:
                    ### satisfy property (1), no bias
                    final_result = original_result
                else:
                    ### progress to step (2)

                    # compute pos_M for male
                    for each_text in sampled_male_mutants[N: N + L]:
                        male_mut_results.append(self.predict(each_text))
                    pos_M = 1.0 * sum(male_mut_results) / (N + L)
                    # compute pos_F for female
                    for each_text in sampled_female_mutants[N: N + L]:
                        female_mut_results.append(self.predict(each_text))
                    pos_F = 1.0 * sum(female_mut_results) / (N + L)

                    ### verify property (2) |pos_M - pos_F| < alpha
                    is_satisfy_prop_2 = True if abs(pos_M - pos_F) < alpha else False
                    ### as long as we proceed to stage 2, we need to repair.
                    final_result = repair_with_majority_rule([original_result] + male_mut_results + female_mut_results)
        consumed_time = time.time() - start_time
        
        if not is_satisfy_prop_2:
            is_bias = True

        if debugging:
            return final_result, is_bias, sampled_male_mutants, sampled_female_mutants, male_mut_results, female_mut_results, consumed_time

        return final_result, is_bias
                
    
    def predict(self, text: str, use_verifier=False):
        '''
        predict the sentiment label of a list of texts
        Parameters
            text:         a piece of text
            use_verifier: specify to use verifier'''

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

