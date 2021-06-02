from utils import nlp
from utils import fnames
from utils import mnames
from Entity import Entity
from Phrase import Phrase
import requests
from Coreference import Coreference


class GenderDecider:
    
    def __init__(self, text):
        self.original = text
        self.docs = nlp(text)
        self.person_entities = None
        self.person_coreferences = None
        self.gender_distribution = None
        self.gender = None



    def getPersonEntities(self) :
        '''
        returns a list of person entities in the text
        '''
        entities = set()
        for ent in self.docs.ents :
            e = Entity(ent.text, ent.start_char, ent.end_char, ent.label_)
            if e.isPerson():
                entities.add(e.getWord())
        return list(entities)
    
    def getPersonCoreferencesAndSingleNames(self):
        '''
        returns a list containing lists of person coreferences and single names which do not have a coreference
        basically gets all the gender words in the text
        '''
        if not self.person_entities:
            self.person_entities = self.getPersonEntities()

        result = self.getPersonCoreferences()
        coreferences = [item.getPhrase() for coref in result for item in coref]
        for e in self.person_entities:
            if e not in coreferences:
                result.append([Phrase(e)])
        
        return result

    def getPersonCoreferences(self) :
        '''
        returns a list containing list of person coreferences
        '''
        coreferences = []
        for r in self.docs._.coref_clusters :
            coref = Coreference(r.main, r.mentions)
            if self.isPersonCoref(coref) : # only take valid coreference
                coreferences.append(coref.getReferences())
        
        return coreferences

    def isPersonCoref(self, coref):
        '''
        returns true/false depending on whether the coreference is a person coreference
        '''
        for phrase in coref.getReferences() :
            if phrase.isGenderPronoun():
                return True
            elif self.isPersonName(phrase.getPhrase()):
                return True
            elif phrase.isContainGenderAssociatedWord() :
                return True
        return False
    
    def isPersonName(self, text) :
        '''
        returns true/false depending on whether the text is a person's name
        '''
        if not self.person_entities:
            self.person_entities = self.getPersonEntities()

        return text in self.person_entities 

    def getGenderDistribution(self) : 
        '''
        returns a dictionary with the gender as the key and a list of words of that gender as the value
        '''

        if not self.person_coreferences:
            self.person_coreferences = self.getPersonCoreferencesAndSingleNames()

        gender_dict = {'male': [],'female': [],'dontknow': []}

        #try to ascertain gender of names using pronouns
        for coref in self.person_coreferences:
            gotGender = False
            gender = ""
            for phrase in coref:
                if phrase.isGenderPronoun():
                    gender = phrase.getGender()
                    gotGender = True
                    break
            
            if gotGender:
                if gender == "male":
                    gender_dict['male'].extend(coref)
                else:
                    gender_dict['female'].extend(coref)
            else:
                gender_dict['dontknow'].extend(coref)

        #names from gender computer
        for phrase in gender_dict['dontknow']:
            word = phrase.getPhrase()
            if word.lower() in fnames:
                gender_dict['female'].append(phrase)
                gender_dict['dontknow'].remove(phrase)
            elif word.lower() in mnames:
                gender_dict['male'].append(phrase)
                gender_dict['dontknow'].remove(phrase)


        #use genderize.io
        for phrase in gender_dict['dontknow']:
            word = phrase.getPhrase()
            response = requests.get(f"https://api.genderize.io?name={word}")
            gender = response.json()["gender"]
            if gender == "female":
                gender_dict['female'].append(phrase)
            else:
                gender_dict['male'].append(phrase)
        
        del gender_dict['dontknow']
        
        return gender_dict
    
    def getGender(self):
        '''
        returns gender of the text using the majority rule
        '''
        if not self.gender_distribution:
            self.gender_distribution = self.getGenderDistribution()
        
        no_of_males = len(self.gender_distribution["male"])
        no_of_females = len(self.gender_distribution["female"])

        if no_of_males == 0 and no_of_females == 0:
            self.gender = "no gender"
        elif no_of_males > no_of_females:
            self.gender = "male"
        elif no_of_males < no_of_females:
            self.gender = "female"
        elif no_of_males == no_of_females:
            self.gender = "cannot decide"

        return self.gender

    def getGenderProportion(self):
        '''
        returns proportion of male words and proportion of female words in the text
        '''

        if not self.gender_distribution:
            self.gender_distribution = self.getGenderDistribution()
        
        no_of_males = len(self.gender_distribution["male"])
        no_of_females = len(self.gender_distribution["female"])
        total = no_of_males + no_of_females

        if total == 0:
            return "no gender"

        male_proportion = no_of_males/total
        female_proportion = no_of_females/total

        return {'Proportion of male words': male_proportion, 'Proportion of female words': female_proportion}





