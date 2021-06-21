#!/usr/bin/env python3


import sqlite3
import csv
import sys

quantize = '--quantize' in sys.argv


with open('datasetSentences.txt', encoding="utf-8") as infile:
    infile.readline()
    sentences = dict(line.strip().split('\t') for line in infile if len(line) > 0)

with open('dictionary.txt', encoding="utf-8") as infile:
    infile.readline()
    dictionary = dict(line.strip().split('|')[::-1] for line in infile if len(line) > 0)

with open('sentiment_labels.txt', encoding="utf-8") as infile:
    infile.readline()
    sentiments = dict(line.strip().strip('!').split('|') for line in infile if len(line) > 0)
    if quantize:
        sentiments = {k: int(float(v) * 5) for k, v in sentiments.items()}

with open('datasetSplit.txt', encoding="utf-8") as infile:
    infile.readline()
    splits = dict(line.strip().split(',') for line in infile if len(line) > 0)


conn = sqlite3.connect(':memory:')
conn.execute('CREATE TABLE sentences (id LONG PRIMARY KEY, sentence TEXT)')
conn.execute('CREATE TABLE dictionary (id LONG PRIMARY KEY, phrase TEXT)')
conn.execute('CREATE TABLE sentiments (phrase_id LONG PRIMARY KEY, sentiment {})'.format('INT' if quantize else 'FLOAT'))
conn.execute('CREATE TABLE splits (sentence_id LONG PRIMARY KEY, partition INT)')

conn.execute('CREATE INDEX sentences_sentence_idx ON sentences (sentence);')
conn.execute('CREATE INDEX dictionary_phrase_idx ON dictionary (phrase);')

conn.executemany('INSERT INTO sentences VALUES (?, ?)', sentences.items())
conn.executemany('INSERT INTO dictionary VALUES (?, ?)', dictionary.items())
conn.executemany('INSERT INTO sentiments VALUES (?, ?)', sentiments.items())
conn.executemany('INSERT INTO splits VALUES (?, ?)', splits.items())

conn.commit()
crs = conn.cursor()

crs.execute('''
        SELECT
            partition,
            sentence,
            sentiment
        FROM sentences 
            JOIN dictionary ON sentence = phrase
            JOIN sentiments ON dictionary.id = phrase_id
            JOIN splits ON sentences.id = sentence_id
        ''')


dataset = crs.fetchall()

for fname, partition in [['train', 1], ['test', 2], ['dev', 3]]:
# for fname, partition in [['test', 1]]:
    with open(fname + '.csv', 'w') as outfile:
        writer = csv.writer(outfile)
        writer.writerow(['sentence', 'sentiment'])
        for row in dataset:
            if row[0] == partition:
                writer.writerow(row[1:])