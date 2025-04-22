# import all necessary libraries
import nltk
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from collections import Counter
ignore(warnings)
# Download necessary NLTK data
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('punkt_tab')  # punkt_tab is optional

# Load corpus
with open("D:\RESEARCH related\PreCog tasks\Language_representations\Data\eng_news_2020_300K\eng_news_2020_300K\eng_news_2020_300K-sentences.txt", "r", encoding='utf8') as f:
    sentences = f.readlines()
    sentences = [sentence.strip() for sentence in sentences]

print(f"Total sentences loaded: {len(sentences)}") # Check the number of sentences loaded

# Clean and tokenize
def preprocess(sentence):
    sentence = sentence.lower()  # Lowercase
    sentence = re.sub(r'[^a-zA-Z\s]', '', sentence)  # Remove punctuations and numbers
    #No, Lemmatization and stemming eliminate subtle meaning differences by reducing word forms
    tokens = word_tokenize(sentence)  # Tokenize using NLTK
    return tokens

# Apply to all sentences and store in a list as processed_sentences
processed_sentences = [preprocess(s) for s in sentences] 
# Example: processed_sentences should be defined beforehand
# processed_sentences = [['the', 'bank', 'approved'], ['river', 'bank', 'was'], ...]

### Build Vocabulary ###

# Step 1: Ask for vocab size
try:
    vocab_size = int(input("Enter the desired vocabulary size (e.g., 10000, 15000, 20000): "))
except ValueError:
    print("Invalid input. Using default vocab size = 15000.")
    vocab_size = 15000

# Step 2: Build vocab based on frequency
flat_words = [word for sentence in processed_sentences for word in sentence]
vocab_counter = Counter(flat_words)
vocab = [word for word, freq in vocab_counter.most_common(vocab_size)]

# Step 3: Ask user for ordering method
print("\nChoose vocabulary ordering method:")
print("1. POS-based order (nouns → verbs → adjectives)")
print("2. Frequency-based order (default)")

choice = input("Enter 1 or 2: ")

# Step 4: Apply ordering
match choice:
    case '1':
        print("\nApplying POS-based ordering...")
        nltk.download('averaged_perceptron_tagger_eng')
        pos_tags = nltk.pos_tag(vocab)

        nouns = [word for word, tag in pos_tags if tag.startswith('NN')]
        verbs = [word for word, tag in pos_tags if tag.startswith('VB')]
        adjs  = [word for word, tag in pos_tags if tag.startswith('JJ')]

        ordered_vocab = nouns + verbs + adjs
        word2id = {word: i for i, word in enumerate(ordered_vocab)}
        id2word = {i: word for i, word in enumerate(ordered_vocab)}

        print(f"Vocabulary reordered by POS: {len(ordered_vocab)} words (nouns+verbs+adjs)")

    case _:
        print("\nUsing frequency-based ordering...")
        word2id = {word: i for i, word in enumerate(vocab)}
        id2word = {i: word for i, word in enumerate(vocab)}
        print(f"Vocabulary size: {len(vocab)}")
