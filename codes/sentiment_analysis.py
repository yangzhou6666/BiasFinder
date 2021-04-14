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

seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

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


    def predict(self, text: str, use_verifier=False):
        '''
        predict the sentiment label of a list of texts
        Parameters
            text:         a piece of text
            use_verifier: specify to use verifier'''
        data_loader = self.convert_text_to_feature([text])
        if use_verifier:
            '''generate mutants use biasfinder'''
            mg = MutantGeneration(text)
            if len(mg.getMutants()) > 0:
                mutant = mg.getMutants()
                data_loader = self.convert_text_to_feature([text] + mutant)



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

        final_result = 1 if sum(results) > len(results) / 2 else 0
        if final_result == 1:
            confidence = sum(results) / (1.0 * len(results))
        else:
            confidence = 1 - sum(results) / (1.0 * len(results))
        
        return final_result, confidence, results

