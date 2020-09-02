# -*- coding: utf-8 -*-
"""
Created on Sat Aug 29 11:58:39 2020

@author: MSI
"""


import pandas as pd

df = pd.read_csv("./new/check.csv")
# df = df.sample(n = 1000, random_state = 123)

excludeList = ['director', 'actor', 'actress', 'filmmaker', 'writer', 'artist', 'scriptwriter', 'reviewer', 'commentator', 'slasher']



from string import punctuation
from string import digits
import spacy
from pycorenlp import StanfordCoreNLP
# nlp_wrapper = StanfordCoreNLP('10.4.0.15:9000')
nlp_wrapper = StanfordCoreNLP('http://localhost:9000')
nlp = spacy.load('en_core_web_sm')

class annotatedText:
    # Class to represent annotated text resulted from StandfordCoreNLP
    
    def __init__(self, text):
    # input:
        # text -> string
        
        self.originalText = text
        self.text = nlp_wrapper.annotate(text,
            properties={
                'ner.applyNumericClassifiers' : 'false',
                'ner.useSUTime' : 'false',
                'annotators': 'ner, pos',
                'outputFormat': 'json',
                'timeout': 9999999999,
            })
    
    def getOriginalText(self):
        # return string of original text -> string
        return self.originalText
    
    def getFullAnnotatedText(self):
        # return the annotated text in json format
        return self.text
    
    def getAnnotatedSentence(self, sentenceIndex):
        # input:
            # sentenceIndex -> integer
        # output:
            # the annotated sentence -> string
        return self.text['sentences'][sentenceIndex]
    
    def getNumberOfSentence(self):
        # output:
            # return the number of sentence in a text -> integer
        return len(self.getFullAnnotatedText()['sentences'])
    
    def getNumberOfToken(self, sentenceIndex):
        # input:
            # sentenceIndex -> integer
        # output:
            # return the number of token in sentence[sentenceIndex] -> integer
        return len(self.getAnnotatedSentence(sentenceIndex)['tokens'])
    
    def getNumberOfEntity(self, sentenceIndex):
        # input:
            # sentenceIndex -> integer
        # output:
            # return the number of entity in integer that exist in a sentence[sentenceIndex] -> integer
        return len(self.getAnnotatedSentence(sentenceIndex)['entitymentions'])
    

class annotatedTextOcc(annotatedText):
    def getName(self, sentenceIndex):
        # input:
            # sentenceIndex -> integer
        # output:
            # return a list of unique name in a sentence[sentenceIndex] if name exist in the sentence, otherwise return empty list
        
        nameList = []
        for i in range(self.getNumberOfEntity(sentenceIndex)):
            tempName = self.getAnnotatedSentence(sentenceIndex)['entitymentions'][i]['text']
            if self.getAnnotatedSentence(sentenceIndex)['entitymentions'][i]['ner'] == 'PERSON' and tempName.lower() not in ['he', 'him', 'his' 'she', 'her', 'they', 'we', 'i', 'you']:
                nameList.append(tempName)
        return nameList
    
    def getOccupation(self, sentenceIndex):
        # input:
            # sentenceIndex -> integer
        # output:
            # return a list of unique occupation in a sentence[sentenceIndex] if occupation exist in the sentence, otherwise return empty list
        occ = None
        offsetBegin = 0
        offsetEnd = 0
        occupationList = []
        for i in range(self.getNumberOfEntity(sentenceIndex)):
            if len(self.getName(sentenceIndex)) > 0:
                if self.getAnnotatedSentence(sentenceIndex)['entitymentions'][i]['ner'] == 'TITLE':
                    temp_occ = self.getAnnotatedSentence(sentenceIndex)['entitymentions'][i]['text']
                    if len(temp_occ.split()) == 1:
                        occ = temp_occ
                        offsetBegin = self.getAnnotatedSentence(sentenceIndex)['entitymentions'][i]['characterOffsetBegin']
                        offsetEnd = self.getAnnotatedSentence(sentenceIndex)['entitymentions'][i]['characterOffsetEnd']
                        occupationList.append((occ, offsetBegin, offsetEnd, sentenceIndex))
        return occupationList
    
    def checkPosTag(self, sentenceIndex, occupation):
        # input:
            # sentenceIndex -> integer
            # occupation -> string
            # occupation must exist in the string, otherwise the function will be invalid
        # output:
            # True if the postag of token is 'NN' (singular noun) or  'NNS' (plural noun)

        if self.getPosTag(sentenceIndex, occupation) == 'NN' or self.getPosTag(sentenceIndex, occupation) == 'NNS':
            return True
        else:
            return False

    def getPosTag(self, sentenceIndex, token):
        # input:
            # token -> string
            # token is assumed always exist in the text
            # sentenceIndex -> integer
        # output:
            # return postag of token -> string
            # if token index is invalid, will return empty string
        
        tokenIndex = self.getTokenIndex(sentenceIndex, token)
        if tokenIndex == -999:
            return ''
        posTag = self.getAnnotatedSentence(sentenceIndex)['tokens'][tokenIndex]['pos']
        return posTag
            
    def getTokenIndex(self, sentenceIndex, token):
        # input:
            # token -> string
            # token is assumed always exist in the text
            # sentenceIndex -> integer
        # output:
            # i -> integer
            # i is index of a token in the sentence
            # if token is not available in the sentence, return -999
        
        for i in range(self.getNumberOfToken(sentenceIndex)):
            if self.getAnnotatedSentence(sentenceIndex)['tokens'][i]['originalText'].strip(punctuation).strip(digits) == token:
                return i
        return -999
    
    def checkDeterminer(self, sentenceIndex, tokenBegin):
        if tokenBegin != 0:
            token = self.getAnnotatedSentence(sentenceIndex)['tokens'][tokenBegin-1]['originalText']
            if token.lower() in ['a', 'an']:
                return True
            else:
                return False
        else:
            return False
        
    # def getTokenEndSentence(self, sentenceIndex):
    #     if sentenceIndex != self.getNumberOfSentence():
    #         # print(self.getNumberOfSentence())
    #         # print(self.getAnnotatedSentence(sentenceIndex))
    #         numberOfToken = (self.getNumberOfToken(sentenceIndex))
    #         # print(numberOfToken)
    #         endToken = self.getAnnotatedSentence(sentenceIndex)['tokens'][numberOfToken-1]['characterOffsetEnd']
    #         beginToken = self.getAnnotatedSentence(sentenceIndex+1)['tokens'][numberOfToken]['characterOffsetEnd']
    #         print(self.getOriginalText())
    #         print(self.getOriginalText()[0:endToken])
    #         print(self.getOriginalText()[beginToken:])
        
import time
start_time = time.time()

counter = 1
templateList1 = []
for index, row in df.iterrows():
    print("counter: {}".format(counter))
    counter += 1
    oriText = row.original
    # oriText = "There must have been a sale on this storyline back in the 40's. An epidemic threatens New York ( it's always New York) and nobody takes it seriously. Some might say that Richard Widmark and Jack Palance did it better in Panic in the Streets, but I disagree. There is always something about these Poverty Row productions that really touch a nerve. The production values are never that polished and the acting is a little rough around the edges, but that is the very reason I think this movie and those like it are effective. Rough, grainy, edgy. And the cast. All 2nd stringers or A list actors past their prime. No egos here. These folks were happy to get the work. Whit Bissell, Carl Benton Reid, Jim Backus, Arthur Space, Charles Korvin, and the melodious voice of Reed Hadley flowing in the background like crude oil. By the way, I've been in the hospital a couple of times; how come my nurses never looked like Dorothy Malone? In these kind of movies they don't bother much with make - up and hair, but they really managed to turn Evelyn Keyes into a hag. Or maybe they just skipped the make - up and hair altogether. Anyway, it was pretty effective. She plays a lovesick jewel smuggler who picks up a case of Small Pox in Cuba while smuggling jewels back for ultra - villain Charles Korvin ( who is boffing her sister in the meantime). You got the Customs Agents looking for her because of the jewels, and the Health Department looking for her because she's about to de - populate New York. No 4th Amendment rights here. Everybody gets hassled. You got ta have the right attitude to enjoy a movie like this. I have a brother who scrutinizes movies to death. If they don't hold up to his Orson Wellian standards, he bombs them unmercifully. They must have the directorial excellence of a David Lean movie, the score of Wolfgang von Korngold, the Sound and Art of Douglas Shearer and Cedric Gibbons respectively. This ain't it. But I have the right attitude, and if you do as well, you'll love this movie."
    at = annotatedTextOcc(oriText)
    occList = []
    for i in range(at.getNumberOfSentence()):
        
        occTempList = at.getOccupation(i)
        
        occList += occTempList
    
    print(occList)
    if len(occList) == 1 and occList[0] not in excludeList:
        templateList1.append((row.template_id,row.template))
        
            
    


print("--- %s seconds ---" % (time.time() - start_time))

# placeholderList = pd.read_csv("./asset/neutral-occupation.csv")
# placeholderList = placeholderList['occupation'].to_list()
# import inflect
# mutantList = []
# p = inflect.engine()
# for t in templateList1:
#     template, label, ori = t
#     for element in placeholderList:
#         mutant = ''.join(template.replace('@DetAAN', (p.a(element).split()[0])).replace('@OCCUPATION', element).replace('@CorefPronounHER', 'her').replace('@CorefPronounHIS', 'his').replace('@CorefPronounSHE', 'she').replace('@CorefPronounHE', 'he').replace("@CorefOCCUPATION", element))
#         mutantList.append((label, mutant, mutant, template, ori, element))



        # if at.checkPosTag(sentenceIndex, occ):
        #     if at.checkDeterminer(sentenceIndex, tokenBegin):
        #         startChunk = oriText[0:offsetBegin-2]
        #         endChunk = oriText[offsetEnd:]
        #         placeholder = '<DET> <OCC>'
        #         template = startChunk + placeholder + endChunk
        #     else:
        #         startChunk = oriText[0:offsetBegin]
        #         endChunk = oriText[offsetEnd:]
        #         placeholder = '<OCC>'
        #         template = startChunk + placeholder + endChunk
            # print(oriText[0:offsetBegin])
        # print(occ)
        # print(oriText[offsetEnd:])
        
        # print()
