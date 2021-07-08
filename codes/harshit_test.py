import sys
sys.path.append('./fine-tuning')
sys.path.append('./gender')

from gender.MutantGeneration import MutantGeneration

# text = "My goodness , Palona has a lot to offer and he seemed to have no problem flaunting his natural gifts ."
text = "My goodness , Palona has a lot to offer and he seemed to have no problem flaunting his natural gifts by himself."

mg = MutantGeneration(text)
print(mg.getTemplate())
print(mg.getConcreteTemplate())

