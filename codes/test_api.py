import pandas as pd
import requests
text = "The Great Dictator is a beyondexcellent film. Charlie Chaplin succeeds in being both extremely funny and witty and yet at the same time provides a strong statement in his satire against fascism. The antiNazi speech by Chaplin at the end, with its values, is one of filmdom's great moments. Throughout this movie, I sensed there was some higher form of intelligence, beyond genuinely intelligent filmmaking, at work."



df = pd.read_csv("../asset/imdb/test.csv", names=["label", "sentence"], sep="\t")
for index, row in df.iterrows():
    label = row["label"]
    text = row["sentence"]
    resp = requests.post("http://127.0.0.1:5000/predict",
                     data={"text": text})

    print(resp.text)