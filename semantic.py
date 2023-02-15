import spacy

nlp = spacy.load('en_core_web_md')
word1 = nlp("cat")
word2 = nlp("monkey")
word3 = nlp("banana")

print(word1.similarity(word2))
print(word3.similarity(word2))
print(word3.similarity(word1))

# Results
# 0.5929930274321619
# 0.40415016164997786
# 0.22358825939615987
# Analysis:
    # cat and monkey have a small delta variance because they are both animals 
    # monkey and banana have small delta variances because we can assume that because monkey eats banana

tokens = nlp('cat apple monkey banana ')
for token1 in tokens:
    for token2 in tokens:
        print(token1.text, token2.text, token1.similarity(token2))
'''
- Cat and monkey seem to be similar because they are both animals;
- Similarly, banana and apple are similar because they are both fruits;
- Interestingly, monkey and banana have a higher similarity than monkey and
    apple. 
- So we can assume that the model already puts together that
    monkeys eat bananas and that is why there is a significant similarity.
- Another interesting fact is that cat does not have any significant similarity
    with any of the fruits although monkey does. 
- So, the model does not
    explicitly seem to recognise transitive relationships in its calculation.
'''

nlp = spacy.load('en_core_web_md')

sentence_to_compare = "Why is my cat on the car"
sentences = ["where did my dog go",
"Hello, there is my car",
"I\'ve lost my car in my car",
"I\'d like my boat back",
"I will name my dog Diana"]

model_sentence = nlp(sentence_to_compare)
for sentence in sentences:
    similarity = nlp(sentence).similarity(model_sentence)
    print(sentence + " - ", similarity)

# Using the "en_core_web_sm" model
nlp = spacy.load('en_core_web_sm')
gardenpathSentences = u"The old man the boat.', 'The complex houses married and single soldiers and their families.','The horse raced past the barn fell.', 'to be led down or up the garden path','The frames are particularly modern in this picture exhibition, because they are made of wood and have been stored in a damp cellar."

doc = nlp(gardenpathSentences)
doc.text.split()
[token.orth_ for token in doc]
print([(token, token.orth_, token.orth) for token in doc])
print([token.orth_ for token in doc if not token.is_punct | token.is_space])
print(spacy.explain("FAC"))
# What was the entity and its explanation that you looked up?
    # The entity i looked up was "picture"
# Did the entity make sense in terms of the word associated with it?
    # It didnt make sense to me though.

