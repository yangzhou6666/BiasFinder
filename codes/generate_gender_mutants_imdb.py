import sys
sys.path.append('./fine-tuning')
sys.path.append('./gender')

import pandas as pd
import numpy as np
import math
import spacy
import os

import time

from gender.utils import preprocessText
from gender.MutantGeneration import MutantGeneration
from multiprocessing import Pool, Process, Queue, Manager
import multiprocessing


def compute_mut():
    '''for multiprocessing uaage'''
    while True:
        if not q.empty():
            index, label, text = q.get()
            mg = MutantGeneration(text)

            if len(mg.getMutants()) > 0:
                original = [text] * len(mg.getMutants())
                label = [label] * len(mg.getMutants())
                concrete_template = [mg.getConcreteTemplate()] * len(mg.getMutants())
                mutant = mg.getMutants()
                gender = mg.getGenders()
                q_to_store.put((
                    index, original, label, concrete_template, mutant, gender
                ))
        else:
            print("Finished")
            return


#df = pd.read_csv("../asset/sst/test.csv", header = 0, sep=",")
df = pd.read_csv("../asset/imdb/test.csv", names = ["label", "sentence"], sep="\t")

start = time.time()


n_template = 0
i = 0
counter = 0

manager = multiprocessing.Manager()

q = manager.Queue()
q_to_store = manager.Queue()


for index, row in df.iterrows():
    # sentiment = row["sentiment"]
    # if sentiment >= 0.5:
    #     label = 1
    # else:
    #     label = 0
    label = row["label"]
    text = row["sentence"]
    q.put((index, label, text))


numList = []
for i in range(8) :
    p = multiprocessing.Process(target=compute_mut, args=())
    numList.append(p)
    p.start()

for i in numList:
    i.join()

print("Generation Process finished.")

### Save in seperated csv files, instead of one.

data_dir = "../data/biasfinder/gender/imdb/mutated_training+5_epoch_fine_tuning+updated_concrete_template_X/each/" 
if not os.path.exists(data_dir) :
    os.makedirs(data_dir)


while not q_to_store.empty():
    ### start to save.

    originals = []
    concrete_templates = []
    mutants = []
    labels = []
    identifiers = []
    types = []
    genders = []
    countries = []

    index, original, label, concrete_template, mutant, gender = q_to_store.get()
    originals.extend(original)
    labels.extend(label)
    concrete_templates.extend(concrete_template)
    mutants.extend(mutant)
    genders.extend(gender)
    
    dm = pd.DataFrame(data={"label": labels, "mutant": mutants, "template": concrete_templates})
    
    dm.to_csv(data_dir + str(index) + '.csv', index=None, header=None, sep="\t")


print("CSV files saved.")
