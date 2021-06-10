from sentiment_analysis import SentimentAnalysis
from flask import Flask, jsonify, request
app = Flask(__name__)


model_checkpoint='./../models/fine-tuning/pytorch_imdb_fine_tuned/epoch5.pt'
bert_config_file='./../models/uncased_L-12_H-768_A-12/bert_config.json'
vocab_file='./../models/uncased_L-12_H-768_A-12/vocab.txt'


### initialize an SA system
sa_system = SentimentAnalysis(model_checkpoint=model_checkpoint,
                            bert_config_file=bert_config_file,
                            vocab_file=vocab_file)

a = 1

@app.route('/predict', methods=['POST'])
def predict():
    result = -1
    if request.method == 'POST':
        mytext = request.form['mytext']
        result, is_bias = sa_system.predict(mytext, True)

    return jsonify({'result': result, 'is_bias': is_bias})


app.run(host='0.0.0.0',port=8887)