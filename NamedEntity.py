from nltk import word_tokenize, pos_tag, ne_chunk
from nltk.chunk import conlltags2tree, tree2conlltags
from nltk.stem.snowball import SnowballStemmer
import os
import string
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

#GMB Corpora
ner_tags = collections.Counter()

corpus_root = "gmb-1.0.0"

for root, dirs, files in os.walk(corpus_root):
    for filename in files:
        if filename.endswith(".tags"):
            with open(os.path.join(root, filename), 'rb') as file_handle:
                file_content = file_handle.read().decode('utf-8').strip()
                annotated = file_content.split('\n\n')
                for annotate in annotated:
                    tokens = [seq for seq in annotate.split('\n') if seq]

                    sftokens = []

                    for idx, tokens in enumerate(tokens):
                        annotations = tokens.split('\t')
                        word, tag, ner = annotations[0], annotations[1], annotations[3]

                        if ner != 'O':
                            ner = ner.split('-')[0]

                        ner_tags[ner] += 1

print("Words = ", sum(ner_tags.values()))
#Words = 82728

