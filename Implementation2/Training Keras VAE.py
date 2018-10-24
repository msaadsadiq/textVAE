from keras.callbacks import ModelCheckpoint
from keras.datasets import imdb
from keras.preprocessing.sequence import pad_sequences
from Keras_VAE import VAE
import numpy as np
import os

MAX_LENGTH = 300
NUM_WORDS = 1000

(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=NUM_WORDS)

print("Training data")
print(X_train.shape)
print(y_train.shape)

print("Number of words:")
print(len(np.unique(np.hstack(X_train))))

X_train = pad_sequences(X_train, maxlen=MAX_LENGTH)
X_test = pad_sequences(X_test, maxlen=MAX_LENGTH)

train_indices = np.random.choice(np.arange(X_train.shape[0]), 2000, replace=False)
test_indices = np.random.choice(np.arange(X_test.shape[0]), 1000, replace=False)

X_train = X_train[train_indices]
y_train = y_train[train_indices]

X_test = X_test[test_indices]
y_test = y_test[test_indices]

temp = np.zeros((X_train.shape[0], MAX_LENGTH, NUM_WORDS))
temp[np.expand_dims(np.arange(X_train.shape[0]), axis=0).reshape(X_train.shape[0], 1),
     np.repeat(np.array([np.arange(MAX_LENGTH)]), X_train.shape[0], axis=0), X_train] = 1

X_train_one_hot = temp

temp = np.zeros((X_test.shape[0], MAX_LENGTH, NUM_WORDS))
temp[np.expand_dims(np.arange(X_test.shape[0]), axis=0).reshape(X_test.shape[0], 1),
     np.repeat(np.array([np.arange(MAX_LENGTH)]), X_test.shape[0], axis=0), X_test] = 1

x_test_one_hot = temp


def create_model_checkpoint(dir, model_name):
    filepath = dir + '/' + \
               model_name + "-{epoch:02d}-{val_decoded_mean_acc:.2f}-{val_pred_loss:.2f}.h5"
    directory = os.path.dirname(filepath)

    try:
        os.stat(directory)
    except:
        os.mkdir(directory)

    checkpointer = ModelCheckpoint(filepath=filepath,
                                   verbose=1,
                                   save_best_only=False)

    return checkpointer


def train():
    model = VAE()
    model.create(vocab_size=NUM_WORDS, max_length=MAX_LENGTH)

    checkpointer = create_model_checkpoint('models', 'rnn_ae')

    model.autoencoder.fit(x=X_train, y={'decoded_mean': X_train_one_hot, 'pred': y_train},
                          batch_size=10, epochs=10, callbacks=[checkpointer],
                          validation_data=(X_test, {'decoded_mean': x_test_one_hot, 'pred':  y_test}))


train()
