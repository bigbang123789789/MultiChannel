from __future__ import print_function

import os
import numpy as np
import tensorflow as tf


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

from keras.models import Model, load_model
from keras.optimizers import Adam
from keras.callbacks import (
    ModelCheckpoint,
    EarlyStopping
)

from keras.preprocessing import sequence

from load_data import load_data_shuffle



np.random.seed(1337)
tf.random.set_seed(1337)



# =========================
# PARAMETERS
# =========================


max_features = 21540

maxlen = 400

batch_size = 32

embedding_dims = 200

nb_filter = 150

hidden_dims = 100

epochs = 20



os.makedirs(
    "CNN-LSTM-weights",
    exist_ok=True
)



# =========================
# MAX POOL
# =========================


def max_1d(x):

    return tf.reduce_max(
        x,
        axis=1
    )



# =========================
# TRAIN
# =========================


cvs = [1]

accs = []



for cv in cvs:


    print("=========================")
    print("Fold:", cv)
    print("=========================")



    X_train, y_train, X_test, y_test, X_val, y_val = load_data_shuffle(cv)



    print(
        len(X_train),
        "train sequences"
    )

    print(
        len(X_test),
        "test sequences"
    )



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



    print(
        "Train:",
        X_train.shape
    )

    print(
        "Val:",
        X_val.shape
    )

    print(
        "Test:",
        X_test.shape
    )



    # =========================
    # MODEL
    # =========================


    input_layer = Input(
        shape=(maxlen,),
        dtype="int32",
        name="input"
    )



    # =========================
    # CNN CHANNEL
    # =========================


    cnn_embedding = Embedding(
        input_dim=max_features,
        output_dim=embedding_dims
    )(input_layer)



    conv3 = Conv1D(
        filters=nb_filter,
        kernel_size=3,
        activation="relu",
        padding="valid"
    )(cnn_embedding)


    pool3 = Lambda(
        max_1d,
        output_shape=(nb_filter,)
    )(conv3)



    conv4 = Conv1D(
        filters=nb_filter,
        kernel_size=4,
        activation="relu",
        padding="valid"
    )(cnn_embedding)


    pool4 = Lambda(
        max_1d,
        output_shape=(nb_filter,)
    )(conv4)



    conv7 = Conv1D(
        filters=nb_filter,
        kernel_size=7,
        activation="relu",
        padding="valid"
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
        128,
        dropout=0.2,
        recurrent_dropout=0.2
    )(lstm_embedding)



    # =========================
    # CNN + LSTM
    # =========================


    merged = Concatenate()(
        [
            cnn_output,
            lstm_output
        ]
    )



    dense = Dense(
        hidden_dims*2,
        activation="relu"
    )(merged)



    dense = Dropout(
        0.5
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
        metrics=[
            "accuracy"
        ]
    )



    model.summary()



    # =========================
    # SAVE BEST MODEL
    # =========================


    checkpoint = ModelCheckpoint(

        filepath=f"CNN-LSTM-weights/cv{cv}.keras",

        monitor="val_accuracy",

        save_best_only=True,

        mode="max",

        verbose=1
    )



    early_stop = EarlyStopping(

        monitor="val_accuracy",

        patience=4,

        restore_best_weights=False,

        mode="max"
    )



    # =========================
    # TRAIN
    # =========================


    history = model.fit(

        X_train,

        y_train,

        batch_size=batch_size,

        epochs=epochs,

        validation_data=(
            X_val,
            y_val
        ),

        callbacks=[
            checkpoint,
            early_stop
        ]

    )



    # =========================
    # LOAD BEST
    # =========================


    print(
        "Loading best model..."
    )


    model = load_model(
        f"CNN-LSTM-weights/cv{cv}.keras"
    )



    # =========================
    # TEST
    # =========================


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



print("=========================")

print(
    "Average Accuracy:",
    np.mean(accs)
)
