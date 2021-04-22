from bias_rv import *
from bias_rv.MutantGeneration import MutantGeneration


text = "The Great Dictator is a beyondexcellent film. Charlie Chaplin succeeds in being both extremely funny and witty and yet at the same time provides a strong statement in his satire against fascism. The antiNazi speech by Chaplin at the end, with its values, is one of filmdom's great moments. Throughout this movie, I sensed there was some higher form of intelligence, beyond genuinely intelligent filmmaking, at work."

mg = MutantGeneration(text)
print(mg.getMutants())