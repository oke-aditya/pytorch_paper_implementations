# Taken from Abhishek Thakur and his kaggle kernel
# https://www.kaggle.com/abhishek/same-old-entity-embeddings/
# https://www.youtube.com/watch?v=EATAM3BOD_E
# Really great implemenation by him and great kernel too.
# Orignal paper https://arxiv.org/abs/1604.06737
# I have modified it to make it generic.
# It should give you a dense representation of categorical columns.

# Please send a PR for PyTorch implemenation.
# Uses tensorflow 2.x. This repo never uses tf 1.x.

import warnings

warnings.filterwarnings("ignore")
import os
import gc
import joblib
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import StratifiedKFold
from sklearn import metrics, preprocessing
from tensorflow.keras import layers
from tensorflow.keras import optimizers
from tensorflow.keras import losses
from tensorflow.keras.models import Model, load_model
from tensorflow.keras import callbacks
from tensorflow.keras import backend as K
from tensorflow.keras import utils

# Just label encoding the data
def preprocess_data(df, id_c, target_c):
    """
    Args:
    df :- Dataframe which should have all categorical variables. Merge train and test beforehand if required.
    id_c :- Unique id column.
    target_c: - Target column
    """
    features = [x for x in df.columns if x not in [id_c, target_c]]

    for feat in features:
        lbl_enc = preprocessing.LabelEncoder()
        df[feat] = lbl_enc.fit_transform(df[feat].fillna("-1").astype(str).values)
        joblib.dump(lbl_enc, f"{feat}_lbl_enc.pkl")
    return df


# Creates the entity embedding Model
def create_model(
    df,
    catcols: list,
    num_classes: int,
    dropout_rate: float = 0.3,
    dense_layers: int = 2,
    dense_dim: int = 300,
):
    """
    Args: 
    df : Categorical Dataframe. Please clean it before.
    catcols : list of columns which are categorical
    num_classes : number of classes of classification.
    dropout_rate : dropout rate of embedding network
    dense_layers : number of dense layers in the netowrk after Concatenating embeddings.
    dense_dim : Number of neurons in the Dense Layer after Concatenating embeddings.
    """
    inputs = []
    outputs = []
    for c in catcols:
        num_unique_values = int(df[c].nunique())
        embed_dim = int(min(np.ceil((num_unique_values) / 2), 50))
        inp = layers.Input(shape=(1,))
        out = layers.Embedding(num_unique_values + 1, embed_dim, name=c)(inp)
        out = layers.SpatialDropout1D(dropout_rate)(out)
        out = layers.Reshape(target_shape=(embed_dim,))(out)
        inputs.append(inp)
        outputs.append(out)

    x = layers.Concatenate()(outputs)
    x = layers.BatchNormalization()(x)

    for i in range(dense_layers):
        x = layers.Dense(dense_dim, activation="relu")(x)
        x = layers.Dropout(dropout_rate)(x)
        x = layers.BatchNormalization()(x)

    y = layers.Dense(num_classes, activation="softmax")(x)

    model = Model(inputs=inputs, outputs=y)
    return model


# Purely optional function you can use any other optimizer and loss function too.
def configure_model(model):
    opt = optimizers.Adam(lr=1e-3)
    loss = losses.BinaryCrossentropy()
    # es = callbacks.EarlyStopping(monitor='val_auc', min_delta=0.001, patience=5,
    #                             verbose=1, mode='max', baseline=None, restore_best_weights=True)

    # rlr = callbacks.ReduceLROnPlateau(monitor='val_auc', factor=0.5,
    #                                 patience=3, min_lr=1e-6, mode='max', verbose=1)

    model.compile(loss=loss, optimizer=opt)
    return model


# if __name__ == "__main__":
# Sample main function to show how it can be used.
# Read the data and get it preprocessed.
# train = pd.read_csv("../input/cat-in-the-dat-ii/train.csv")
# test = pd.read_csv("../input/cat-in-the-dat-ii/test.csv")
# sample = pd.read_csv("../input/cat-in-the-dat-ii/sample_submission.csv")
# test["target"] = -1
# data = pd.concat([train, test]).reset_index(drop=True)

# data = preprocess_data(data, "id", "target")

# # Seprating the train and test back
# features = [x for x in train.columns if x not in ["id", "target"]]
# train = data[data.target != -1].reset_index(drop=True)
# test = data[data.target == -1].reset_index(drop=True)
# test_data = [test.loc[:, features].values[:, k] for k in range(test.loc[:, features].values.shape[1])]

# Split the data into test and train and simply call fit.

# model.fit(X_train,
#   utils.to_categorical(y_train),
#   validation_data=(X_test, utils.to_categorical(y_test)),
#   verbose=1,
#   batch_size=1024,
#   callbacks=[es, rlr],
#   epochs=100
#  )

# Just do a model.fit when you need it.
