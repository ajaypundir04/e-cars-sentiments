from nltk.corpus import wordnet

def get_synonyms(word):
    synonyms = []
    for syn in wordnet.synsets(word):
        for lemma in syn.lemmas():
            synonyms.append(lemma.name())
    return set(synonyms)



if __name__ == "__main__":
    word = "charging"
    related_words = get_synonyms(word)
    print(related_words)



#Problem (identification of wrong features/words from urls)
  # our system will generate similar words on the basis of a seed word (filter the data)[biased]
  # we provide some fixed set of words(skip) [non-biased](identification of words) 
    #(biasing)
    
# Motivation ()    
    # LLM(pre-tune)(small langugae model) (fine-tune with ev)(LLAMA3.2 (open-source))
      # sentiments (analysis by model)
    
# Respond Survey
    # Distribute among the Univ of Munster [20] [Scale]       
    # Professor will help me with sharing it to Norway [10]    
    # one Chinese Person( my ex-colleague) [2] 
     # fill the survey on the basis of articles () [help of LLM]
    
 #LLM (some-knowlegde) (not a human response)
    # knowledge graphs (fine tuning of LLAMA) 
    # various articles (pros and cons of ev's)
    # evaulation of articles by LLAMA
