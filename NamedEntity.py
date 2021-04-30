from nltk import word_tokenize, pos_tag, ne_chunk
from nltk.chunk import conlltags2tree, tree2conlltags
import os
import collections

sentence = "You stinky boy who never showers"

#NER CHUNKING
ne_tree = (ne_chunk(pos_tag(word_tokenize(sentence))))
#[OUTPUT] (S You/PRP stinky/VBP boy/NNS who/WP never/RB showers/NNS)

#IOB TAGGING
iob_tagged = tree2conlltags(ne_tree)
print(iob_tagged)
#[OUTPUT] [('You', 'PRP', 'O'), ('stinky', 'VBP', 'O'), ('boy', 'NNS', 'O'), ('who', 'WP', 'O'), ('never', 'RB', 'O'), ('showers', 'NNS', 'O')]

ne_tree = conlltags2tree(iob_tagged)
print(ne_tree)
#[OUTPUT] (S You/PRP stinky/VBP boy/NNS who/WP never/RB showers/NNS)

