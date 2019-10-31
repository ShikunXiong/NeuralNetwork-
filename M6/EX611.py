import numpy as np
import string
from keras.preprocessing.text import Tokenizer

if __name__ == "__main__":
    samples = ['The cat sat on the mat.', 'The dog ate my homework.']

    token_index = {}
    for sample in samples:
        for word in sample.split():
            if word not in token_index:
                token_index[word] = len(token_index) + 1

    max_length = 10
    results = np.zeros((len(samples), max_length, max(token_index.values()) + 1))
    for i, sample in enumerate(samples):
        for j, word in list(enumerate(sample.split()))[:max_length]:
            index = token_index.get(word)
            results[i, j, index] = 1.

    samples = ['The cat sat on the mat.', 'The dog ate my homework.']
    characters = string.printable  # All printable ASCII characters.
    token_index = dict(zip(characters, range(1, len(characters) + 1)))

    max_length = 50
    results = np.zeros((len(samples), max_length, max(token_index.values()) + 1))
    for i, sample in enumerate(samples):
        for j, character in enumerate(sample[:max_length]):
            index = token_index.get(character)
            results[i, j, index] = 1.

    tokenizer = Tokenizer(num_words=1000)
    tokenizer.fit_on_texts(samples)
    sequences = tokenizer.texts_to_sequences(samples)

    one_hot_results = tokenizer.texts_to_matrix(samples, mode='binary')

    word_index = tokenizer.word_index
    print('Found %s unique tokens.' % len(word_index))