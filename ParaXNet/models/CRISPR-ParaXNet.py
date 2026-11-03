# -*- coding: utf-8 -*-
"""
Created on Mon Aug 21 18:05:18 2023

@author: xincao
"""
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import Input, InputLayer, Dense, Conv2D, Flatten, BatchNormalization, MaxPooling2D, Dropout
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import os
import scipy.stats as st
from scipy.stats import gaussian_kde
from scipy.stats import spearmanr
from scipy.stats import pearsonr
from scipy import stats
from scipy.stats import bootstrap
print(tf.keras.backend.image_data_format())
from sklearn.preprocessing import PowerTransformer
import tensorflow_recommenders as tfrs
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from tensorflow.keras.callbacks import Callback
import time

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_CSV = os.path.join(os.path.dirname(HERE), 'data', 'crisprSQL.csv')
df= pd.read_csv(DATA_CSV,low_memory=False,na_values=['NA', '', ' '])
dna = list(df['target_sequence'].to_numpy())
grna = list(df['grna_target_sequence'].to_numpy())
label = list(df['log_label'].to_numpy())

def encoding(target_seq, grna_seq):
    
    dict1 = {'A': [1, 0, 0, 0, 0],'T': [0, 1, 0, 0, 0], 
             'G': [0, 0, 1, 0, 0],'C': [0, 0, 0, 1, 0],
             '-': [0, 0, 0, 0, 1]}
    dict2 = {'A':5, 'G':4, 'C':3, 'T':2, '-':1}
     
    def grna_encode(grna_single_seq):
        code_list = []
        for i in range(len(grna_single_seq)):
            code_list.append(dict1[grna_single_seq[i]])
        return code_list
    
    def dna_encode(target_single_seq):
        code_list = []
        for i in range(len(target_single_seq)):
            code_list.append(dict1[target_single_seq[i]])
        return code_list

    or_code = []
    dir_code = np.zeros(2)
    single_seq = []
    result1 = []
    result2 = []
    
    for ii in range(len(target_seq)):
        for jj in range(len(target_seq[ii])):
            temp1 = (np.bitwise_or(dna_encode(target_seq[ii][jj]), grna_encode(grna_seq[ii][jj]))).squeeze()   
            if dict2[target_seq[ii][jj]] == dict2[grna_seq[ii][jj]]:
                pass
            else:
                if dict2[target_seq[ii][jj]] > dict2[grna_seq[ii][jj]]:
                    dir_code[0] = 1
                else:
                    dir_code[1] = 1
            temp2 = np.concatenate((temp1,dir_code))       
            temp2 = np.array(temp2)
            single_seq.append(temp2)
            
            if jj == len(target_seq[ii])-1:
                single_seq = np.array(single_seq)
                result1.append(single_seq)
                single_seq = []   
            
    result2 = np.array(result1)
    
    return result2 

X = encoding(dna,grna)
Y = np.array(label).reshape(-1,1)
X = np.asarray(X).astype('float32')
Y = np.asarray(Y).astype('float32')
xt = X.transpose(0,2,1)
del xt, dna, grna, label

inputRow = 7
inputCol = 23
X=X
Y=Y

indices = np.arange(Y.shape[0])

# Two-stage split: 80% train+validation / 20% test, then split the
# train+validation subset into 80% train / 20% validation.
# Final proportions: 64% train / 16% validation / 20% test.
train_val_indices, test_indices = train_test_split(
    indices, test_size=0.2, random_state=42, shuffle=True
)
train_indices, val_indices = train_test_split(
    train_val_indices, test_size=0.2, random_state=42, shuffle=True
)
print(
    f"Split sizes - train: {len(train_indices)}, "
    f"validation: {len(val_indices)}, test: {len(test_indices)}"
)

df_train = df.loc[train_indices].copy()
df_val = df.loc[val_indices].copy()
df_test = df.loc[test_indices].copy()
xtrain = X[train_indices]
ytrain = Y[train_indices]
xval = X[val_indices]
yval = Y[val_indices]
xtest = X[test_indices]
ytest = Y[test_indices]

def transformImages(
        xtrain, xval, xtest,
        ytrain, yval, ytest,
        imgrows, imgcols,
                    ):
    if tf.keras.backend.image_data_format() == 'channels_first':
        xtrain = xtrain.reshape(xtrain.shape[0], imgrows * imgcols)
        xval = xval.reshape(xval.shape[0], imgrows * imgcols)
        xtest = xtest.reshape(xtest.shape[0], imgrows * imgcols)
        input_shape = (1, imgrows, imgcols)
    else:
        xtrain = xtrain.reshape(xtrain.shape[0], imgrows * imgcols)
        xval = xval.reshape(xval.shape[0], imgrows * imgcols)
        xtest = xtest.reshape(xtest.shape[0], imgrows * imgcols)
        input_shape = (imgrows * imgcols)
    xtrain = xtrain.astype('float32')
    xval = xval.astype('float32')
    xtest = xtest.astype('float32')
    return xtrain, xval, xtest, ytrain, yval, ytest, input_shape
xtrain, xval, xtest, ytrain, yval, ytest, input_shape = transformImages(
    xtrain, xval, xtest, ytrain, yval, ytest, inputRow, inputCol
)
X2 = X.transpose(0,2,1)
X = X2.reshape((len(X), inputRow*inputCol))
#%% 
loss = []
validation_loss = []
pearson = []
spearman = []
# epoch number (EN); batch size (BS); learning rate (LR)
EN = 100
BS = 200
LR = 0.0001

def Model():    
    input_layer = Input(shape=[input_shape], name='main_input')
    x = Dense(128, activation='relu', kernel_initializer="uniform")(input_layer)
    x = BatchNormalization()(x)
    x = Dropout(0.1)(x)
    x = Dense(128, activation='relu', kernel_initializer="uniform")(x)
    x = Dense(128, activation='relu', kernel_initializer="uniform")(x)
    
    embedding_layer = input_layer
    cross_layer = input_layer
    for _ in range(3):
        cross_layer = tfrs.layers.dcn.Cross()(embedding_layer, cross_layer)
    x = tf.keras.layers.concatenate([cross_layer, x], name = 'concatenate_all_features')   
    x = Dense(128, activation='relu', kernel_initializer="uniform")(x)
    x = Dropout(0.1)(x)
    output_layer = Dense(1, activation='linear', kernel_initializer="uniform", name='main_output')(x)
    return keras.Model(inputs=input_layer, outputs=output_layer)

model = Model()
model.summary()
model.compile(
            optimizer=keras.optimizers.Adam(learning_rate = LR),
            loss = 'mse',
            metrics=['mse']
            )
hist = model.fit(xtrain, ytrain,
                     batch_size = BS,
                     epochs = EN,
                     validation_data = (xval, yval),
                     verbose = 1,
                     )

loss.append(hist.history['loss'])
validation_loss.append([hist.history['val_loss']])
predictions = model.predict(xtest)
coef, p = spearmanr(np.squeeze(ytest), np.squeeze(predictions))
corr, p1 = pearsonr(np.squeeze(ytest), np.squeeze(predictions))
pearson.append(corr)
spearman.append(coef)

plt.figure()
plt.plot(hist.history['loss'], label='Train Loss')
plt.plot(hist.history['val_loss'], label='Validation Loss')
plt.title('FNN5 Training Curves')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

#%%
df_test['pred'] = predictions
pred_logback = np.exp(predictions) - 1.0
df_test['pred_logback'] = pred_logback

def select_top_percent(df, column_name='pred'):
    top_n = max(1, int(len(df) * 0.05))
    return df.nlargest(top_n, column_name)
top_percent_df = select_top_percent(df_test)

from typing import Sequence
from sklearn import metrics

def cumulative_true(
    y_true: Sequence[float],
    y_pred: Sequence[float]
) -> np.ndarray:

  df = pd.DataFrame({
      'y_true': y_true,
      'y_pred': y_pred,
  }).sort_values(
      by='y_pred', ascending=False)

  return (df['y_true'].cumsum() / df['y_true'].sum()).values


def gini_from_gain(df: pd.DataFrame) -> pd.DataFrame:
  raw = df.apply(lambda x: 2 * x.sum() / df.shape[0] - 1.)
  normalized = raw / raw[0]
  return pd.DataFrame({
      'raw': raw,
      'normalized': normalized
  })[['raw', 'normalized']]

gain_ground_truth = cumulative_true(top_percent_df['cleavage_freq'], top_percent_df['cleavage_freq'])
gain_model = cumulative_true(top_percent_df['cleavage_freq'], top_percent_df['pred_logback'])
gain = pd.DataFrame({
    'ground_truth': gain_ground_truth,
    'model': gain_model
})
gini = gini_from_gain(gain)
print(gini)

#%%
r2 = r2_score(ytest, predictions)
mae = mean_absolute_error(ytest, predictions)
mape = np.mean(np.abs((ytest - predictions) / ytest)) * 100
rmse = np.sqrt(mean_squared_error(ytest, predictions))
mse = mean_squared_error(ytest, predictions)
print()
print(f"Pearson: {pearson}")
print(f"Spearman: {spearman}")
print()
print(f"R²: {r2:.4f}")
print(f"MSE: {mse:.4f}")
print(f"MAE: {mae:.4f}")
print(f"RMSE: {rmse:.4f}")

#%%
import pandas as pd
import numpy as np

def select_top_n_percent(df, column, top_n_percent):
    
    sorted_df = df.sort_values(by=column, ascending=False)
    
    num_elements = len(df)
    top_n_count = max(int(np.ceil((top_n_percent / 100) * num_elements)), 1)  
    
    top_df = sorted_df.head(top_n_count)
    
    top_values = top_df[column].values
    top_indices = top_df.index.values  
    
    return top_values, top_indices


top5_label, top5_indices = select_top_n_percent(df_test, "cleavage_freq", 5)
top1_label, top1_indices = select_top_n_percent(df_test, "cleavage_freq", 1)

top1_pred = df_test.loc[top1_indices, 'pred_logback'].values
top1_01label = df_test.loc[top1_indices, '01_label'].values

top5_pred = df_test.loc[top5_indices, 'pred_logback'].values
top5_01label = df_test.loc[top5_indices, '01_label'].values




top5_MAPE = np.mean(np.abs((top5_label - top5_pred) / top5_label)) * 100
top1_MAPE = np.mean(np.abs((top1_label - top1_pred) / top1_label)) * 100


print("Top 5% MAPE:", top5_MAPE)
print("Top 1% MAPE:", top1_MAPE)