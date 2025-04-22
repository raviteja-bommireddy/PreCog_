"""
co_occurrence_matrix.py

Constructs a co-occurrence matrix from a list of tokenized sentences (processed_sentences),
based on a sliding context window.

Usage:
- Ensure `processed_sentences`, `word2id`, and `vocab_size` are defined before importing this.
- Adjust `window_size` to experiment with syntactic vs. semantic context capture.

Author: Your Name
"""

import numpy as np
from tqdm import tqdm

def build_cooccurrence_matrix(processed_sentences, word2id, vocab_size, window_size=4):
    """
    Builds a co-occurrence matrix using a sliding window over tokenized text.

    Args:
        processed_sentences (list): List of tokenized sentences (e.g., [['the', 'cat'], ['sat', 'there']])
        word2id (dict): Mapping of words to their integer index.
        vocab_size (int): Number of words in the vocabulary.
        window_size (int): Size of context window to the left and right of target word.

    Returns:
        np.ndarray: Co-occurrence matrix of shape (vocab_size, vocab_size)
    """
    print(f"\nBuilding co-occurrence matrix with window size = {window_size}...")
    cooc_matrix = np.zeros((vocab_size, vocab_size), dtype=np.float32)

    for sentence in tqdm(processed_sentences, desc="Processing sentences"):
        # Keep only words in vocab
        sentence = [word for word in sentence if word in word2id]
        for idx, target_word in enumerate(sentence):
            target_id = word2id[target_word]
            start = max(0, idx - window_size)
            end = min(len(sentence), idx + window_size + 1)
            for context_pos in range(start, end):
                if context_pos != idx:
                    context_word = sentence[context_pos]
                    context_id = word2id[context_word]
                    cooc_matrix[target_id][context_id] += 1

    print("Co-occurrence matrix construction complete.")
    return cooc_matrix

print("\nco-occurrence matrix will be created with these codes in notebooks.")
print('''
1. window_size = 4  # You can change this based on your needs
2. cooc_matrix = build_cooccurrence_matrix(processed_sentences, word2id, vocab_size, window_size)
      ''')