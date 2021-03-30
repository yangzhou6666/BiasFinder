import sys
sys.path.append('./fine-tuning')

from modeling_single_layer import BertConfig, BertForSequenceClassification
import torch
import tokenization
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm, trange
import numpy as np


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

    def convert_text_to_feature(self, text, max_seq_length=128):
        '''
        Convert text to features that BERT can take in
        '''
        # 1: convert to unicode
        text = tokenization.convert_to_unicode(text)

        # TODO 2: convert to features
        # eval_features = convert_examples_to_features(eval_examples, label_list, args.max_seq_length, tokenizer, trunc_medium=args.trunc_medium)
        tokens_a = self.tokenizer.tokenize(text)

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
        ## 但貌似并不能设置为-1
        feature = InputFeatures(
                        input_ids=input_ids,
                        input_mask=input_mask,
                        segment_ids=segment_ids,
                        label_id=label_id)

        tensor_data = TensorDataset(torch.tensor([input_ids], dtype=torch.long), 
                                torch.tensor([input_mask], dtype=torch.long), 
                                torch.tensor([segment_ids], dtype=torch.long), 
                                torch.tensor([label_id], dtype=torch.long))
        
        data_loader = DataLoader(tensor_data, batch_size=1, shuffle=False)

        return data_loader


    def predict(self, text: str, use_verifier=False):
        '''
        predict the sentiment label of a list of texts
        Parameters
            text:         a piece of text
            use_verifier: specify to use verifier'''
        
        # TODO: covert text: str to features that bert can take in
        data_loader = self.convert_text_to_feature(text)

        self._model.eval()

        device = self.device

        for input_ids, input_mask, segment_ids, label_ids in tqdm(data_loader, desc="Evaluate"):
            input_ids = input_ids.to(device)
            input_mask = input_mask.to(device)
            segment_ids = segment_ids.to(device)
            label_ids = label_ids.to(device)
            
            with torch.no_grad():
                logits = self._model(input_ids=input_ids, token_type_ids=segment_ids, attention_mask=input_mask, labels=None)
            logits = logits.detach().cpu().numpy()
            print(logits)
            predicted_label = np.argmax(logits, axis=1)
            print(predicted_label)







if __name__=="__main__":
    model_checkpoint='./../models/fine-tuning/pytorch_imdb_fine_tuned/epoch5.pt'
    bert_config_file='./../models/uncased_L-12_H-768_A-12/bert_config.json'
    vocab_file='./../models/uncased_L-12_H-768_A-12/vocab.txt'

    ### initialize an SA system
    sa_system = SentimentAnalysis(model_checkpoint=model_checkpoint,
                                bert_config_file=bert_config_file,
                                vocab_file=vocab_file)
    for _ in range(10):
        sa_system.predict("I am happy")




