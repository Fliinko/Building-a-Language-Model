from nltk import word_tokenize, pos_tag
import nltk 
#nltk.download('treebank')

pos_tag(word_tokenize('I love Joseph, the next one, eighty thousand'))
#[OUTPUT] [('I', 'PRP'), ('love', 'VBP'), ('Joseph', 'NNP'), (',', ','), ('the', 'DT'), ('next', 'JJ'), ('one', 'NN'), (',', ','), ('eighty', 'VBZ'), ('thousand', 'NN')] 

tagged = nltk.corpus.treebank.tagged_sents()
print(tagged[0])
#[OUTPUT] [('Pierre', 'NNP'), ('Vinken', 'NNP'), (',', ','), ('61', 'CD'), ('years', 'NNS'), ('old', 'JJ'), (',', ','), ('will', 'MD'), ('join', 'VB'), ('the', 'DT'), ('board', 'NN'), ('as', 'IN'), ('a', 'DT'), ('nonexecutive', 'JJ'), ('director', 'NN'), ('Nov.', 'NNP'), ('29', 'CD'), ('.', '.')
print("Tagged Sentences: ", len(tagged)) #3914
print("Tagged Words: ", len(nltk.corpus.treebank.tagged_words())) #100676

def features(sentence, index):
    """ sentence: [w1, w2, ...], index: the index of the word """
    return {
        'word': sentence[index],
        'is_first': index == 0,
        'is_last': index == len(sentence) - 1,
        'is_capitalized': sentence[index][0].upper() == sentence[index][0],
        'is_all_caps': sentence[index].upper() == sentence[index],
        'is_all_lower': sentence[index].lower() == sentence[index],
        'prefix-1': sentence[index][0],
        'prefix-2': sentence[index][:2],
        'prefix-3': sentence[index][:3],
        'suffix-1': sentence[index][-1],
        'suffix-2': sentence[index][-2:],
        'suffix-3': sentence[index][-3:],
        'prev_word': '' if index == 0 else sentence[index - 1],
        'next_word': '' if index == len(sentence) - 1 else sentence[index + 1],
        'has_hyphen': '-' in sentence[index],
        'is_numeric': sentence[index].isdigit(),
        'capitals_inside': sentence[index][1:].lower() != sentence[index][1:]
    }
 
import pprint 
pprint.pprint(features(['This', 'is', 'a', 'sentence'], 2))
 
{'capitals_inside': False,
 'has_hyphen': False,
 'is_all_caps': False,
 'is_all_lower': True,
 'is_capitalized': False,
 'is_first': False,
 'is_last': False,
 'is_numeric': False,
 'next_word': 'sentence',
 'prefix-1': 'a',
 'prefix-2': 'a',
 'prefix-3': 'a',
 'prev_word': 'is',
 'suffix-1': 'a',
 'suffix-2': 'a',
 'suffix-3': 'a',
 'word': 'a'}
 
def untag(tagged):
    return [w for w, t in tagged]

#Splitting the Dataset into Test and Train
cutoff = int(.75 * len(tagged))
training = tagged[:cutoff]
test = tagged[cutoff:]

print(len(training))
print(len(test))

def dataset(tagged):
    x, y = [], []

    for t in tagged:
        for index in range(len(t)):
            x.append(features(untag(t), index))
            y.append(t[index][1])

    return x, y

x, y = dataset(training)

from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_extraction import DictVectorizer
from sklearn.pipeline import Pipeline

clf = Pipeline([('vectorizer', DictVectorizer(sparce=False)),
                ('classifier', DecisionTreeClassifier(criterion='entropy'))])

clf.fit(x[:10000], y[:10000])
print('Training Complete.')

x_test, y_test = dataset(test)
print('Accuracy:' ,clf.score(x_test, y_test))

def pos_tag(sentence):
    tags = clf.predict([features(sentence, index) for index in range(len(sentence))])
    return zip(sentence, tags)

print(pos_tag(word_tokenize('Monkey makes funny noises and I laugh')))
