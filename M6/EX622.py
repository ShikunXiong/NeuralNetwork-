from keras.layers import LSTM
from keras.models import Sequential
from keras.layers import Embedding
from keras.datasets import imdb
from keras.layers import Dense
import matplotlib.pyplot as plt
from keras.preprocessing import sequence

if __name__ == "__main__":
    max_features = 10000
    maxlen = 500
    model = Sequential()
    (input_train, y_train), (input_test, y_test) = imdb.load_data(num_words=max_features)
    model.add(Embedding(max_features, 32))
    model.add(LSTM(32))
    model.add(Dense(1, activation='sigmoid'))

    input_train = sequence.pad_sequences(input_train, maxlen=maxlen)
    input_test = sequence.pad_sequences(input_test, maxlen=maxlen)

    model.compile(optimizer='rmsprop',
                  loss='binary_crossentropy',
                  metrics=['acc'])
    history = model.fit(input_train, y_train,
                        epochs=10,
                        batch_size=128,
                        validation_split=0.2)
    acc = history.history['acc']
    val_acc = history.history['val_acc']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs = range(len(acc))

    plt.plot(epochs, acc, 'bo', label='Training acc')
    plt.plot(epochs, val_acc, 'b', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.legend()

    plt.figure()

    plt.plot(epochs, loss, 'bo', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()

    plt.show()