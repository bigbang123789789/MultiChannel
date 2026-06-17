from __future__ import print_function

import os
import numpy as np

from keras.layers import (
    Dense,
    Dropout,
    Lambda,
    Input,
    Embedding,
    LSTM,
    Conv1D,
    Concatenate
)

from keras.models import Model
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint

from keras.preprocessing import sequence
import tensorflow as tf

from load_data import load_data_shuffle


np.random.seed(1337)


# =========================
# PARAMETERS
# =========================

max_features = 21540

maxlen = 400

batch_size = 10

embedding_dims = 200

nb_filter = 150

hidden_dims = 100

epochs = 14



if not os.path.exists("CNN-LSTM-weights"):
    os.makedirs("CNN-LSTM-weights")



# =========================
# MAX POOL
# =========================

def max_1d(x):
    return tf.reduce_max(
        x,
        axis=1
    )



# =========================
# TRAIN 5 FOLD
# =========================

cvs = [1]

accs = []


for cv in cvs:


    print("Loading data for cv...", cv)



    X_train, y_train, X_test, y_test, X_val, y_val = load_data_shuffle(cv)



    print(len(X_train), "train sequences")
    print(len(X_test), "test sequences")



    X_train = sequence.pad_sequences(
        X_train,
        maxlen=maxlen
    )


    X_val = sequence.pad_sequences(
        X_val,
        maxlen=maxlen
    )


    X_test = sequence.pad_sequences(
        X_test,
        maxlen=maxlen
    )



    print("X_train shape:", X_train.shape)
    print("X_val shape:", X_val.shape)
    print("X_test shape:", X_test.shape)



    print("Build model...")



    # =========================
    # INPUT
    # =========================


    input_layer = Input(
        shape=(maxlen,),
        dtype="int32",
        name="main_input"
    )



    # =========================
    # CNN CHANNEL
    # =========================


    cnn_embedding = Embedding(
        input_dim=max_features,
        output_dim=embedding_dims
    )(input_layer)



    # filter 3


    conv3 = Conv1D(
        filters=nb_filter,
        kernel_size=3,
        padding="valid",
        activation="relu"
    )(cnn_embedding)



    pool3 = Lambda(
        max_1d,
        output_shape=(nb_filter,)
    )(conv3)





    # filter 4


    conv4 = Conv1D(
        filters=nb_filter,
        kernel_size=4,
        padding="valid",
        activation="relu"
    )(cnn_embedding)



    pool4 = Lambda(
        max_1d,
        output_shape=(nb_filter,)
    )(conv4)





    # filter 7


    conv7 = Conv1D(
        filters=nb_filter,
        kernel_size=7,
        padding="valid",
        activation="relu"
    )(cnn_embedding)



    pool7 = Lambda(
        max_1d,
        output_shape=(nb_filter,)
    )(conv7)



    cnn_output = Concatenate()(
        [
            pool3,
            pool4,
            pool7
        ]
    )



    # =========================
    # LSTM CHANNEL
    # =========================


    lstm_embedding = Embedding(
        input_dim=max_features,
        output_dim=embedding_dims
    )(input_layer)



    lstm_output = LSTM(
        128
    )(lstm_embedding)



    # =========================
    # CNN + LSTM
    # =========================


    merged = Concatenate()(
        [
            lstm_output,
            cnn_output
        ]
    )



    dense = Dense(
        hidden_dims*2,
        activation="sigmoid"
    )(merged)



    dense = Dropout(
        0.2
    )(dense)



    output = Dense(
        3,
        activation="softmax"
    )(dense)



    model = Model(
        inputs=input_layer,
        outputs=output
    )



    model.compile(
        loss="categorical_crossentropy",
        optimizer=Adam(
            learning_rate=0.001
        ),
        metrics=["accuracy"]
    )



    model.summary()



    checkpoint = ModelCheckpoint(
        filepath=f"CNN-LSTM-weights/cv{cv}_weights.weights.h5",
        monitor="val_accuracy",
        save_best_only=True,
        mode="max",
        verbose=1
    )



    # =========================
    # TRAIN
    # =========================


    model.fit(
        X_train,
        y_train,
        batch_size=batch_size,
        epochs=epochs,
        validation_data=(X_val,y_val),
        callbacks=[checkpoint]
    )



    # =========================
    # TEST
    # =========================


    model.load_weights(
        f"CNN-LSTM-weights/cv{cv}_weights.weights.h5",
        skip_mismatch=True
    )


    score, acc = model.evaluate(
        X_test,
        y_test,
        batch_size=batch_size
    )


    print(
        "CV",
        cv,
        "Accuracy:",
        acc
    )


    accs.append(acc)



print(
    "========================="
)

print(
    "Average Accuracy:",
    np.mean(accs)
) 
