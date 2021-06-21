import sys
sys.path.append('./gender')
from gender.MutantGeneration import MutantGeneration


test = ["A somewhat crudely constructed but gripping , questing look at a person so racked with self-loathing , he becomes an enemy to his own race .", 
"It is interesting and fun to see Palona himself and his chimpanzees on the bigger-than-life screen.",
"Using his audience as a figurative port-of-call , Palona pulls his even-handed ideological ship to their dock for unloading , before he continues his longer journey still ahead ."
]
for text in test:
    mg = MutantGeneration(text)
    print("Template: ",mg.getTemplate())
    print("Concrete Template: ",mg.getConcreteTemplate())
    print("")



