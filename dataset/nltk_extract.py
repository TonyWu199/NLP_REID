import nltk
import re
import pprint
import os
from nltk import Tree
from bert import sent2words

sentence = "The man is visible from the rear.  He is wearing a white tee shirt top with short sleeves.  His shorts are gray.  He is carrying something in his right hand. His shoes are black and white."
# text = "A pedestrian with dark hair is wearing red and white shoes, a black hooded sweatshirt, and black pants."
# text = "The person has short black hair and is wearing black pants, a long sleeve black top, and red sneakers."
# text = "A woman wearing a black sleeveless shirt, a pair of blue jean shorts and a pair of pink shoes."
# The man wears a light colored shirt with short sleeves and tan-colored shorts as well as sandals.
# A woman slants forward wearing a loose black, halter-neck top swinging in front of her body over a cottony, white skirt ending above her knees and thick sandals.
# A woman looks over to her right, has hair hanging over her front shoulders, carries a brown shoulder bag over her right shoulder with her right arm in front of it and extends her right leg in back of her. She wears a black jacket with wide light-gray horizontal stripe over the front and sleeves, light-blue jeans and white shoes with black trim.
# A man with black hair, wearing a black t-shirt, and dark shorts is walking and is carrying a knapsack over both shoulders.

# parse the tokens and extract the Noun Phrases
def nltk_NPS(words):
    patterns = """
        NP:    
           {<DT><WP><VBP>*<RB>*<VBN><IN><NN>}
           {<NN|NNS|NNP|NNPS><IN>*<NN|NNS|NNP|NNPS>+}
           {<JJ>*<NN|NNS|NNP|NNPS><CC>*<NN|NNS|NNP|NNPS>+}
           {<JJ>*<NN|NNS|NNP|NNPS>+}
            """
    NPChunker = nltk.RegexpParser(patterns)
    tree = NPChunker.parse(nltk.pos_tag(words))
    print(tree)
    # print(tree)
    nps = []
    for subtree in tree.subtrees():
        if subtree.label() == 'NP':
            np = subtree
            np = ' '.join(word for word, tag in np.leaves())
            nps.append(np)
    return nps

#standford tagger
from nltk.tag import StanfordPOSTagger
os.environ['JAVAHOME'] = '/usr/lib/jvm/java-11-openjdk-amd64'
stf_eng_tagger = StanfordPOSTagger(
                    model_filename='/data/wuziqiang/stanford-postagger-full-2015-12-09/models/english-bidirectional-distsim.tagger',
                    path_to_jar='/data/wuziqiang/stanford-postagger-full-2015-12-09/stanford-postagger.jar',
                    java_options='-mx2g'
                )
def stanford_NPS(words):
    pos_tag = stf_eng_tagger.tag(words)
    print(pos_tag)

sentence = sentence.lower()
words = nltk.word_tokenize(sentence)
nps = nltk_NPS(words)
# stanford_NPS(words)
# print(nps)