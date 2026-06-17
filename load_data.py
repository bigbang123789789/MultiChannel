import numpy as np
from sklearn.utils import shuffle


def load_data_shuffle(cv):


    base = (
        "data/VSFC/Data_not_token/Fold_"
        + str(cv)
        + "/"
    )


    # ==================
    # LOAD
    # ==================

    pos_train = np.load(base+"train_pos.npy")
    neg_train = np.load(base+"train_neg.npy")
    neu_train = np.load(base+"train_neu.npy")


    pos_test = np.load(base+"test_pos.npy")
    neg_test = np.load(base+"test_neg.npy")
    neu_test = np.load(base+"test_neu.npy")



    # ==================
    # LABEL
    # ==================

    y_pos_train = np.tile(
        [1,0,0],
        (len(pos_train),1)
    )


    y_neg_train = np.tile(
        [0,1,0],
        (len(neg_train),1)
    )


    y_neu_train = np.tile(
        [0,0,1],
        (len(neu_train),1)
    )



    y_pos_test = np.tile(
        [1,0,0],
        (len(pos_test),1)
    )


    y_neg_test = np.tile(
        [0,1,0],
        (len(neg_test),1)
    )


    y_neu_test = np.tile(
        [0,0,1],
        (len(neu_test),1)
    )



    # ==================
    # MERGE TRAIN
    # ==================

    X = np.concatenate(
        [
            pos_train,
            neg_train,
            neu_train
        ]
    )


    y = np.concatenate(
        [
            y_pos_train,
            y_neg_train,
            y_neu_train
        ]
    )


    # shuffle toàn bộ

    X, y = shuffle(
        X,
        y,
        random_state=42
    )



    # ==================
    # VALIDATION SPLIT
    # ==================

    split = int(
        0.9 * len(X)
    )


    X_train = X[:split]

    y_train = y[:split]


    X_val = X[split:]

    y_val = y[split:]



    # ==================
    # TEST
    # ==================

    X_test = np.concatenate(
        [
            pos_test,
            neg_test,
            neu_test
        ]
    )


    y_test = np.concatenate(
        [
            y_pos_test,
            y_neg_test,
            y_neu_test
        ]
    )


    X_test, y_test = shuffle(
        X_test,
        y_test,
        random_state=42
    )



    return (
        X_train,
        y_train,
        X_test,
        y_test,
        X_val,
        y_val
    )
