from sentiment_analysis import SentimentAnalysis


if __name__=="__main__":


    model_checkpoint='./../models/fine-tuning/pytorch_imdb_fine_tuned/epoch5.pt'
    bert_config_file='./../models/uncased_L-12_H-768_A-12/bert_config.json'
    vocab_file='./../models/uncased_L-12_H-768_A-12/vocab.txt'

    ### initialize an SA system
    sa_system = SentimentAnalysis(model_checkpoint=model_checkpoint,
                                bert_config_file=bert_config_file,
                                vocab_file=vocab_file)
    
    text = "The Great Dictator is a beyondexcellent film. Charlie Chaplin succeeds in being both extremely funny and witty and yet at the same time provides a strong statement in his satire against fascism. The antiNazi speech by Chaplin at the end, with its values, is one of filmdom's great moments. Throughout this movie, I sensed there was some higher form of intelligence, beyond genuinely intelligent filmmaking, at work."

    result = sa_system.predict(text, True)
    print(result)