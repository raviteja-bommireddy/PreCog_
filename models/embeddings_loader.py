import gensim
import numpy as np

# Path to the pretrained Google News Word2Vec model
model_path = 'D:/RESEARCH related/PreCog tasks/Language_representations/models/GoogleNews-vectors-negative300.bin.gz'

# Load the Word2Vec model
model = gensim.models.KeyedVectors.load_word2vec_format(model_path, binary=True)

# Path to save the numpy file
save_path = 'D:/RESEARCH related/PreCog tasks/Language_representations/models/word2vec.npy'

# Extract the word vectors
word_vectors = model.vectors  # This is a NumPy array of shape (vocabulary_size, vector_size)

# Save the word vectors to a .npy file
np.save(save_path, word_vectors)

# Print a message to confirm the file has been saved
print(f"Word2Vec model saved as {save_path}")


# Load the word vectors from the saved .npy file
word_vectors = np.load('D:/RESEARCH related/PreCog tasks/Language_representations/models/word2vec.npy')

# Check the shape of the loaded word vectors
print(word_vectors.shape)
# Output: (3000000, 300) or similar depending on the model