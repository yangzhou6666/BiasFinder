import pandas as pd
from sentiment_analysis import SentimentAnalysis
import time

if __name__=="__main__":


    model_checkpoint='./../models/fine-tuning/pytorch_imdb_fine_tuned/epoch5.pt'
    bert_config_file='./../models/uncased_L-12_H-768_A-12/bert_config.json'
    vocab_file='./../models/uncased_L-12_H-768_A-12/vocab.txt'

    ### initialize an SA system
    sa_system = SentimentAnalysis(model_checkpoint=model_checkpoint,
                                bert_config_file=bert_config_file,
                                vocab_file=vocab_file)
    
    text = "The Great Dictator is a beyondexcellent film. Charlie Chaplin succeeds in being both extremely funny and witty and yet at the same time provides a strong statement in his satire against fascism. The antiNazi speech by Chaplin at the end, with its values, is one of filmdom's great moments. Throughout this movie, I sensed there was some higher form of intelligence, beyond genuinely intelligent filmmaking, at work."


    df = pd.read_csv("../asset/imdb/test.csv", names=["label", "sentence"], sep="\t")
    count = 0
    correct_count_RV = 0
    correct_count = 0
    not_best_case = 0

    start = time.time()
    for index, row in df.iterrows():
        count += 1
        if count % 1000 == 0:
            pass
        label = row["label"]
        text = row["sentence"]
        final_result, confidence, results = sa_system.predict(text, True)

        # compute SA system accuracy
        if (int(label) == final_result):
            correct_count_RV += 1

        if confidence != 1:
            print((int(label) == final_result))
            not_best_case += 1
        
        final_result, confidence, results = sa_system.predict(text, False)
        if (int(label) == final_result):
            correct_count += 1

    
    print("Accuracy after RV: ", 1.0 * correct_count_RV / count)
    print("Accuracy without RV: ", 1.0 * correct_count / count)
    print("Potential bias rate: ", 1.0 * not_best_case / count)

    end = time.time()

    print("Time consumed: ", end - start)