import datasets
from datasets import load_dataset
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification
from torch.optim import AdamW
from transformers import get_scheduler
import torch
from tqdm.auto import tqdm
import evaluate
import random
import argparse
from nltk.corpus import wordnet, stopwords
from nltk import word_tokenize, pos_tag
from nltk.tokenize.treebank import TreebankWordDetokenizer

random.seed(0)


def example_transform(example):
    example["text"] = example["text"].lower()
    return example


### Rough guidelines --- typos
# For typos, you can try to simulate nearest keys on the QWERTY keyboard for some of the letter (e.g. vowels)
# You can randomly select each word with some fixed probability, and replace random letters in that word with one of the
# nearest keys on the keyboard. You can vary the random probablity or which letters to use to achieve the desired accuracy.


### Rough guidelines --- synonym replacement
# For synonyms, use can rely on wordnet (already imported here). Wordnet (https://www.nltk.org/howto/wordnet.html) includes
# something called synsets (which stands for synonymous words) and for each of them, lemmas() should give you a possible synonym word.
# You can randomly select each word with some fixed probability to replace by a synonym.

def convert_pos(input):
    if input.startswith('J'):
        return wordnet.ADJ
    elif input.startswith("V"):
        return wordnet.VERB
    elif input.startswith("N"):
        return wordnet.NOUN
    elif input.startswith("R"):
        return wordnet.ADV

def custom_transform(example):
    ################################
    ##### YOUR CODE BEGINGS HERE ###

    # Design and implement the transformation as mentioned in pdf
    # You are free to implement any transformation but the comments at the top roughly describe
    # how you could implement two of them --- synonym replacement and typos.

    # You should update example["text"] using your transformation

    protected = {
        # Negations
        "not", "no", "never", "none", "neither",
        "nor", "nothing", "nobody", "without",

        # Contractions
        "n't", "dont", "doesnt", "didnt",
        "isnt", "wasnt", "cant", "wont",

        # Intensifiers
        "very", "extremely", "really", "absolutely",
        "completely", "totally", "barely", "hardly",

        # Contrast words
        "but", "however", "although", "though",
        "despite", "unless", "except"
    }

    protected = protected.union(set(stopwords.words("english")))

    words = example["text"].split(" ")
    tagged = pos_tag(words)
    eligible_indices = [
        i for i, word in enumerate(words)
        if word.lower() not in protected and word.isalpha()
    ]

    swap_amount = int(len(eligible_indices) * 0.5)
    for i in random.sample(eligible_indices, swap_amount):
        synset = wordnet.synsets(words[i], pos=convert_pos(tagged[i][1]))
        if not synset:
            continue

        original_word = words[i].casefold()
        synonyms = [
            lemma.name() for lemma in synset[0].lemmas()
            if lemma.name().casefold() != original_word
        ]
        if synonyms:
            words[i] = random.choice(synonyms)

    candidates = []

    for i, word in enumerate(words):
        if len(word) < 6 or not word.isalpha() or word.lower() in protected:
            continue

        # Swap different interior letters so every selected word changes.
        positions = [j for j in range(1, len(word) - 2) if word[j] != word[j + 1]]
        if positions:
            candidates.append((i, positions))

    for i, positions in random.sample(candidates, int(0.5 * len(candidates))):
        word = words[i]
        j = random.choice(positions)
        words[i] = word[:j] + word[j + 1] + word[j] + word[j + 2:]

    example["text"] = " ".join(words)


    ##### YOUR CODE ENDS HERE ######


    return example
