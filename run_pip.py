import sys
sys.path.append('./codes/fine-tuning')

from codes.bias_rv.BiasRV import biasRV

import pandas as pd
from codes.sentiment_analysis import SentimentAnalysis


model_checkpoint='./models/fine-tuning/pytorch_imdb_fine_tuned/epoch5.pt'
bert_config_file='./models/uncased_L-12_H-768_A-12/bert_config.json'
vocab_file='./models/uncased_L-12_H-768_A-12/vocab.txt'


### initialize an SA system
sa_system = SentimentAnalysis(model_checkpoint=model_checkpoint,
                            bert_config_file=bert_config_file,
                            vocab_file=vocab_file)

df = pd.read_csv("./asset/imdb/test.csv", names=["label", "sentence"], sep="\t")

rv = biasRV(sa_system.predict,X=4,Y=26,alpha=0.1)

count = 0
biased_prediction = []
count_set = [371, 3744, 3962, 6442, 11730, 13953, 14490, 15048, 16421, 18661, 19715, 21328, 21581, 22620, 23395]

for index, row in df.iterrows():
    count += 1
    # if not count in count_set:
    #     continue
    label = row["label"]
    text = row["sentence"]
    result, is_bias = rv.verify_only_property_2(text)
    # result, is_bias = rv.verify(text)
    if not is_bias:
        print(count)
        biased_prediction.append(count)

print(biased_prediction)
print(len(biased_prediction))