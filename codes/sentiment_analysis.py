import sys
sys.path.append('./fine-tuning')

from modeling_single_layer import BertConfig, BertForSequenceClassification
import torch
import tokenization


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

        ### Initialize a bert model
        bert_config = BertConfig.from_json_file(bert_config_file)
        self._model = BertForSequenceClassification(bert_config, len([0,1]), [11, 10], pooling=None)
        self._model.bert.load_state_dict(torch.load(model_checkpoint, map_location='cpu'))
        device = torch.device("cuda")
        self._model.to(device)

        ### Initialize tokenizer
        self.tokenizer = tokenization.FullTokenizer(vocab_file=vocab_file, do_lower_case=True)



    def predict(self, text: [str], use_verifier=False):
        '''
        predict the sentiment label of a list of texts
        Parameters
            text:         list of texts
            use_verifier: specify to use verifier'''
        pass

if __name__=="__main__":
    model_checkpoint='./../models/fine-tuning/pytorch_imdb_fine_tuned/epoch5.pt'
    bert_config_file='./../models/uncased_L-12_H-768_A-12/bert_config.json'
    vocab_file='./../models/uncased_L-12_H-768_A-12/vocab.txt'

    ### initialize an SA system
    sa_system = SentimentAnalysis(model_checkpoint=model_checkpoint,
                                bert_config_file=bert_config_file,
                                vocab_file=vocab_file)



