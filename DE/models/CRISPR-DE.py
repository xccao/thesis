# -*- coding: utf-8 -*-
"""
Created on Wed Aug 30 11:56:36 2023

@author: xincao
"""

import os

import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
# from keras.models import Model
from tensorflow.keras.layers import Input, InputLayer, Dense, Conv2D, Flatten, BatchNormalization, MaxPooling2D, Dropout, Attention
from tensorflow.keras.layers import Reshape, LSTM, Bidirectional, Permute
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import scipy.stats as st
from scipy.stats import gaussian_kde
from scipy.stats import spearmanr
from scipy.stats import pearsonr
from scipy import stats
from scipy.stats import bootstrap
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor


print(tf.keras.backend.image_data_format())
tf.keras.utils.set_random_seed(42)

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DATA_CSV = os.path.join(ROOT, 'data', 'crisprSQL.csv')
RESULT_DIR = os.path.join(ROOT, 'results')
os.makedirs(RESULT_DIR, exist_ok=True)

df= pd.read_csv(DATA_CSV,low_memory=False,na_values=['NA', '', ' '])

# df = pd.read_csv('/Users/tencentgo/Documents/github/crisprDBA/July_new/16634step2.csv')
# df = pd.read_csv(r'C:\2024_workspace\all_in_one\16558_32.csv')

dna = list(df['target_sequence'].to_numpy())
grna = list(df['grna_target_sequence'].to_numpy())
label = list(df['log_label'].to_numpy())
# label = list(df['boxcox'].to_numpy())


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
                # single_seq = np.array(single_seq).reshape(-1, 161) 
                single_seq = np.array(single_seq)
                # single_seq = np.rot90(single_seq,1)
                # single_seq = np.flip(single_seq,axis=0)
                result1.append(single_seq)
                single_seq = []   
                
    # result2 = np.array(result1).reshape(-1,23,7)
    result2 = np.array(result1)
    
    return result2 # result2 size 7*23, change to 23*7 by comment line 78 & 79


X = encoding(dna,grna)
Y = np.array(label).reshape(-1,1)
X = np.asarray(X).astype('float32')
Y = np.asarray(Y).astype('float32')
xt = X.transpose(0,2,1)

# del xt, dna, grna, label

inputRow = 7
inputCol = 23
X=X
Y=Y
# X = X.reshape((len(X),1,inputCol,inputRow))
X_rnn = X.reshape((len(X),1,inputCol,inputRow))

# xtrain, xtest, ytrain, ytest = train_test_split(X, Y, test_size=0.2, random_state=42, shuffle=True)
indices = np.arange(Y.shape[0])







temp_indices, test_indices = train_test_split(indices, test_size=0.1, random_state=42)
train_indices, val_indices = train_test_split(temp_indices, test_size=1/9, random_state=42)

df_train = df.loc[train_indices]
df_test = df.loc[test_indices]
df_val = df.loc[val_indices]

xtrain = X[train_indices]
ytrain = Y[train_indices]
xtest = X[test_indices]
ytest = Y[test_indices]
xval = X[val_indices]
yval = Y[val_indices]

# Dedicated validation inputs: validation is used for model.fit; test remains held out.
xval_fnn = xval.reshape(len(xval), inputRow * inputCol).astype('float32')
xval_cnn = xval.reshape(len(xval), inputRow, inputCol, 1).astype('float32')

xtrain_rnn = X_rnn[train_indices]
xtest_rnn = X_rnn[test_indices]
xval_rnn = X_rnn[val_indices]



# # Split the indices into training and testing sets
# train_indices, test_indices = train_test_split(indices, test_size=0.2, random_state=42)

# df_train = df.loc[train_indices]
# df_test = df.loc[test_indices]
# # Use these indices to index into your data (optional)
# xtrain = X[train_indices]
# ytrain = Y[train_indices]
# xtest = X[test_indices]
# ytest = Y[test_indices]

# xtrain_rnn = X_rnn[train_indices]
# xtest_rnn = X_rnn[test_indices]

#%% FNN pure
def transformImages(
        xtrain, xtest,
        ytrain, ytest,
        imgrows, imgcols,
                    ):
    if tf.keras.backend.image_data_format() == 'channels_first':
        xtrain = xtrain.reshape(xtrain.shape[0], imgrows * imgcols)
        xtest = xtest.reshape(xtest.shape[0], imgrows * imgcols)
        input_shape = (1, imgrows, imgcols)
    else:
        xtrain = xtrain.reshape(xtrain.shape[0], imgrows * imgcols)
        xtest = xtest.reshape(xtest.shape[0], imgrows * imgcols)
        input_shape = (imgrows * imgcols)
    xtrain = xtrain.astype('float32')
    xtest = xtest.astype('float32')
    return xtrain, xtest, ytrain, ytest, input_shape
xtrain, xtest, ytrain, ytest, input_shape = transformImages(xtrain,xtest,ytrain,ytest,inputRow,inputCol)
# X2 = X.transpose(0,2,1)
# X = X2.reshape((len(X), inputRow*inputCol))
xtest_fnn = xtest

loss_fnn = []
validation_loss_fnn = []
pearson_fnn = []
spearman_fnn = []
# epoch number (EN); batch size (BS); learning rate (LR)
EN_fnn = 70
BS_fnn = 400
LR_fnn = 0.0001

def FNN():    
    input_layer = Input(shape=input_shape, name='main_input')
    x = Dense(50, activation='relu', kernel_initializer="uniform")(input_layer)
    # x = BatchNormalization()(x)
    # x = Dropout(0.1)(x)
    # x = Dense(75, activation='relu', kernel_initializer="uniform")(x)
    # x = BatchNormalization()(x)
    x = Dense(50, activation='relu', kernel_initializer="uniform")(x)
    # x = BatchNormalization()(x)
    # x = Dropout(0.1)(x)
    # x = Dense(25, activation='relu', kernel_initializer="uniform")(x)
    
    x = Dense(50, activation='relu', kernel_initializer="uniform")(x)
    x = Dropout(0.1)(x)
    output_layer = Dense(1, activation='linear', kernel_initializer="uniform", name='main_output')(x)
    return keras.Model(inputs=input_layer, outputs=output_layer)

model_fnn = FNN()
model_fnn.summary()
model_fnn.compile(
            optimizer=keras.optimizers.Adam(learning_rate = LR_fnn),
            loss = 'mse',
            metrics=[
                # tf.keras.metrics.AUC(name='roc_auc', curve='ROC'),
                # tf.keras.metrics.AUC(name='pr_auc', curve='PR'),
                # tf.keras.metrics.Precision(name='precision'),
                # tf.keras.metrics.Recall(name='recall'),
                # tf.keras.metrics.BinaryAccuracy(name='accuracy')
                    ]
            )
hist_fnn = model_fnn.fit(xtrain, ytrain,
                     batch_size = BS_fnn,
                     epochs = EN_fnn,
                     validation_data = (xval_fnn, yval),
                     verbose = 1 )#, callbacks=[es]) 

loss_fnn.append(hist_fnn.history['loss'])
validation_loss_fnn.append([hist_fnn.history['val_loss']])
predictions_fnn = model_fnn.predict(xtest_fnn)
coef, p = spearmanr(np.squeeze(ytest), np.squeeze(predictions_fnn))
corr, p1 = pearsonr(np.squeeze(ytest), np.squeeze(predictions_fnn))
pearson_fnn.append(corr)
spearman_fnn.append(coef)

plt.figure()
plt.plot(hist_fnn.history['loss'], label='Train Loss')
plt.plot(hist_fnn.history['val_loss'], label='Validation Loss')
plt.title('FNN Training Curves')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

r2_fnn = r2_score(ytest, predictions_fnn)
mae_fnn = mean_absolute_error(ytest, predictions_fnn)
mape_fnn = np.mean(np.abs((ytest - predictions_fnn) / ytest)) * 100
rmse_fnn = np.sqrt(mean_squared_error(ytest, predictions_fnn))
mse_fnn = mean_squared_error(ytest, predictions_fnn)

print()
print(f"Pearson: {pearson_fnn}")
print(f"Spearman: {spearman_fnn}")
print(f"R²: {r2_fnn:.4f}")
print(f"MSE: {mse_fnn:.4f}")
print(f"MAE: {mae_fnn:.4f}")
print(f"RMSE: {rmse_fnn:.4f}")
print(f"MAPE: {mape_fnn:.2f}%")
#%% CNN pure
def transformImages(
        xtrain, xtest,
        ytrain, ytest,
        imgrows, imgcols,
                    ):
    if tf.keras.backend.image_data_format() == 'channels_first':
        xtrain = xtrain.reshape(xtrain.shape[0], 1, imgrows, imgcols)
        xtest = xtest.reshape(xtest.shape[0], 1, imgrows, imgcols)
        input_shape = (1, imgrows, imgcols)
    else:
        xtrain = xtrain.reshape(xtrain.shape[0], imgrows, imgcols, 1)
        xtest = xtest.reshape(xtest.shape[0], imgrows, imgcols, 1)
        input_shape = (imgrows, imgcols, 1)
    xtrain = xtrain.astype('float32')
    xtest = xtest.astype('float32')
    return xtrain, xtest, ytrain, ytest, input_shape
xtrain, xtest, ytrain, ytest, input_shape = transformImages(xtrain,xtest,ytrain,ytest,inputRow,inputCol)
xtest_cnn = xtest

loss_cnn = []
validation_loss_cnn = []
pearson_cnn = []
spearman_cnn = []
# epoch number (EN); batch size (BS); learning rate (LR) 
EN_cnn = 50
BS_cnn = 200
LR_cnn = 0.0001

def CNN():    
    input_layer = Input(shape=(input_shape), name='main_input')
    x = Dense(100, activation='relu', kernel_initializer="uniform")(input_layer) #100
    x = Conv2D(32,(3,3),activation='relu',kernel_initializer="uniform",padding='SAME')(x)#8
    x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size=(3, 3))(x)
    x = Conv2D(32,(3,3),activation='relu',kernel_initializer="uniform",padding='SAME')(x)#8
    x = Conv2D(32,(3,3),activation='relu',kernel_initializer="uniform",padding='SAME')(x)#8
    x = Flatten()(x)
    x = Dense(100, activation='relu',kernel_initializer="uniform")(x)#100
    x = Dropout(rate=0.1)(x) 
    output_layer = Dense(1, activation='linear', kernel_initializer="uniform", name='main_output')(x)
    return keras.Model(inputs=input_layer, outputs=output_layer)

model_cnn = CNN()
model_cnn.summary()
model_cnn.compile(
            optimizer=keras.optimizers.Adam(learning_rate = LR_cnn),
            loss = 'mse',
            # metrics=['mse']
            metrics=[
                # tf.keras.metrics.AUC(name='roc_auc', curve='ROC'),
                # tf.keras.metrics.AUC(name='pr_auc', curve='PR'),
                # tf.keras.metrics.Precision(name='precision'),
                # tf.keras.metrics.Recall(name='recall'),
                # tf.keras.metrics.BinaryAccuracy(name='accuracy')
                    ]
            )
hist_cnn = model_cnn.fit(xtrain, ytrain,
                     batch_size = BS_cnn,
                     epochs = EN_cnn,
                     validation_data = (xval_cnn, yval),
                     verbose = 1 )#, callbacks=[es]) 

loss_cnn.append(hist_cnn.history['loss'])
validation_loss_cnn.append([hist_cnn.history['val_loss']])
predictions_cnn = model_cnn.predict(xtest_cnn)
coef, p = spearmanr(np.squeeze(ytest), np.squeeze(predictions_cnn))
corr, p1 = pearsonr(np.squeeze(ytest), np.squeeze(predictions_cnn))
pearson_cnn.append(corr)
spearman_cnn.append(coef)

plt.figure()
plt.plot(hist_cnn.history['loss'], label='Train Loss')
plt.plot(hist_cnn.history['val_loss'], label='Validation Loss')
plt.title('CNN Training Curves')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

r2_cnn = r2_score(ytest, predictions_cnn)
mae_cnn = mean_absolute_error(ytest, predictions_cnn)
mape_cnn = np.mean(np.abs((ytest - predictions_cnn) / ytest)) * 100
rmse_cnn = np.sqrt(mean_squared_error(ytest, predictions_cnn))
mse_cnn = mean_squared_error(ytest, predictions_cnn)

print()
print(f"Pearson: {pearson_cnn}")
print(f"Spearman: {spearman_cnn}")
print(f"R²: {r2_cnn:.4f}")
print(f"MSE: {mse_cnn:.4f}")
print(f"MAE: {mae_cnn:.4f}")
print(f"RMSE: {rmse_cnn:.4f}")
print(f"MAPE: {mape_cnn:.2f}%")
#%% RNN-LSTM pure
loss_rnn = []
validation_loss_rnn = []
pearson_rnn = []
spearman_rnn = []
# epoch number (EN); batch size (BS); learning rate (LR)
EN_rnn = 50
BS_rnn = 200
LR_rnn = 0.0001

def RNN():    
    input_layer = Input(shape=(1,23,7), name='main_input')
    x = layers.Conv2D(16, (1, 3), strides=1, padding='same', use_bias=True,trainable=True)(input_layer)
    x = layers.BatchNormalization(axis=-1,scale=True)(x)
    x = Reshape((23, 16))(x)
    x = Bidirectional(LSTM(10, return_sequences=True, input_shape=(23, 16), name="LSTM_out"))(x)
    x = Flatten()(x)
    x = Dense(100, activation='relu')(x)
    # x = Dense(50, activation='relu')(x)
    x = Dropout(0.1)(x)
    output_layer = Dense(1, activation='linear', name='main_output')(x)
    return keras.Model(inputs=input_layer, outputs=output_layer)

model_rnn = RNN()
model_rnn.summary()
model_rnn.compile(
            optimizer=keras.optimizers.Adam(learning_rate = LR_rnn),
            loss = 'mse',
            metrics=[
                # tf.keras.metrics.AUC(name='roc_auc', curve='ROC'),
                # tf.keras.metrics.AUC(name='pr_auc', curve='PR'),
                # tf.keras.metrics.Precision(name='precision'),
                # tf.keras.metrics.Recall(name='recall'),
                # tf.keras.metrics.BinaryAccuracy(name='accuracy')
                    ]
            )
hist_rnn = model_rnn.fit(xtrain_rnn, ytrain,
                     batch_size = BS_rnn,
                     epochs = EN_rnn,
                     validation_data = (xval_rnn, yval),
                     verbose = 1 )#, callbacks=[es]) 

loss_rnn.append(hist_rnn.history['loss'])
validation_loss_rnn.append([hist_rnn.history['val_loss']])
predictions_rnn = model_rnn.predict(xtest_rnn)
coef, p = spearmanr(np.squeeze(ytest), np.squeeze(predictions_rnn))
corr, p1 = pearsonr(np.squeeze(ytest), np.squeeze(predictions_rnn))
pearson_rnn.append(corr)
spearman_rnn.append(coef)

plt.figure()
plt.plot(hist_rnn.history['loss'], label='Train Loss')
plt.plot(hist_rnn.history['val_loss'], label='Validation Loss')
plt.title('RNN Training Curves')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

r2_rnn = r2_score(ytest, predictions_rnn)
mae_rnn = mean_absolute_error(ytest, predictions_rnn)
mape_rnn = np.mean(np.abs((ytest - predictions_rnn) / ytest)) * 100
rmse_rnn = np.sqrt(mean_squared_error(ytest, predictions_rnn))
mse_rnn = mean_squared_error(ytest, predictions_rnn)

print()
print(f"Pearson: {pearson_rnn}")
print(f"Spearman: {spearman_rnn}")
print(f"R²: {r2_rnn:.4f}")
print(f"MSE: {mse_rnn:.4f}")
print(f"MAE: {mae_rnn:.4f}")
print(f"RMSE: {rmse_rnn:.4f}")
print(f"MAPE: {mape_rnn:.2f}%")

#%% attention pure
loss_attention = []
validation_loss_attention = []
pearson_attention = []
spearman_attention = []
# epoch number (EN); batch size (BS); learning rate (LR)
EN_attention = 50
BS_attention = 100
LR_attention = 0.0001

def attention():    
    input_layer = Input(shape=(1,23,7), name='main_input')
    x = layers.Conv2D(16, (1, 3), strides=1, padding='same', use_bias=True,trainable=True)(input_layer)
    x = layers.BatchNormalization(axis=-1,scale=True)(x)
    x = Reshape((23, 16))(x)
    x = Bidirectional(LSTM(10, return_sequences=True, input_shape=(23, 16), name="LSTM_out"))(x)
    x = Attention()([x, x])
    x = Flatten()(x)
    x = Dense(100, activation='relu')(x)
    # x = Dense(50, activation='relu')(x)
    x = Dropout(0.1)(x)
    output_layer = Dense(1, activation='linear', name='main_output')(x)
    return keras.Model(inputs=input_layer, outputs=output_layer)

model_attention = attention()
model_attention.summary()
model_attention.compile(
            optimizer=keras.optimizers.Adam(learning_rate = LR_attention),
            loss = 'mse',
            metrics=[
                # tf.keras.metrics.AUC(name='roc_auc', curve='ROC'),
                # tf.keras.metrics.AUC(name='pr_auc', curve='PR'),
                # tf.keras.metrics.Precision(name='precision'),
                # tf.keras.metrics.Recall(name='recall'),
                # tf.keras.metrics.BinaryAccuracy(name='accuracy')
                    ]
            )
hist_attention = model_attention.fit(xtrain_rnn, ytrain,
                     batch_size = BS_attention,
                     epochs = EN_attention,
                     validation_data = (xval_rnn, yval),
                     verbose = 1 )#, callbacks=[es]) 

loss_attention.append(hist_attention.history['loss'])
validation_loss_attention.append([hist_attention.history['val_loss']])
predictions_attention = model_attention.predict(xtest_rnn)
coef, p = spearmanr(np.squeeze(ytest), np.squeeze(predictions_attention))
corr, p1 = pearsonr(np.squeeze(ytest), np.squeeze(predictions_attention))
pearson_attention.append(corr)
spearman_attention.append(coef)

plt.figure()
plt.plot(hist_attention.history['loss'], label='Train Loss')
plt.plot(hist_attention.history['val_loss'], label='Validation Loss')
plt.title('attention Training Curves')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

r2_attention = r2_score(ytest, predictions_attention)
mae_attention = mean_absolute_error(ytest, predictions_attention)
mape_attention = np.mean(np.abs((ytest - predictions_attention) / ytest)) * 100
rmse_attention = np.sqrt(mean_squared_error(ytest, predictions_attention))
mse_attention = mean_squared_error(ytest, predictions_attention)

print()
print(f"Pearson: {pearson_attention}")
print(f"Spearman: {spearman_attention}")
print(f"R²: {r2_attention:.4f}")
print(f"MSE: {mse_attention:.4f}")
print(f"MAE: {mae_attention:.4f}")
print(f"RMSE: {rmse_attention:.4f}")
print(f"MAPE: {mape_attention:.2f}%")


#%% mcd FNN pure
def transformImages(
        xtrain, xtest,
        ytrain, ytest,
        imgrows, imgcols,
                    ):
    if tf.keras.backend.image_data_format() == 'channels_first':
        xtrain = xtrain.reshape(xtrain.shape[0], imgrows * imgcols)
        xtest = xtest.reshape(xtest.shape[0], imgrows * imgcols)
        input_shape = (1, imgrows, imgcols)
    else:
        xtrain = xtrain.reshape(xtrain.shape[0], imgrows * imgcols)
        xtest = xtest.reshape(xtest.shape[0], imgrows * imgcols)
        input_shape = (imgrows * imgcols)
    xtrain = xtrain.astype('float32')
    xtest = xtest.astype('float32')
    return xtrain, xtest, ytrain, ytest, input_shape
xtrain, xtest, ytrain, ytest, input_shape = transformImages(xtrain,xtest,ytrain,ytest,inputRow,inputCol)
# X2 = X.transpose(0,2,1)
# X = X2.reshape((len(X), inputRow*inputCol))
xtest_fnn = xtest

loss_mcd = []
validation_loss_mcd = []
pearson_mcd = []
spearman_mcd = []
# epoch number (EN); batch size (BS); learning rate (LR)
EN_mcd = 70
BS_mcd = 400
LR_mcd = 0.0001

def mcd():    
    input_layer = Input(shape=input_shape, name='main_input')
    x = Dense(50, activation='relu', kernel_initializer="uniform")(input_layer)
    # x = BatchNormalization()(x)
    x = Dropout(0.05)(x)
    # x = Dense(75, activation='relu', kernel_initializer="uniform")(x)
    # x = BatchNormalization()(x)
    x = Dense(50, activation='relu', kernel_initializer="uniform")(x)
    # x = BatchNormalization()(x)
    # x = Dropout(0.05)(x)
    # x = Dense(25, activation='relu', kernel_initializer="uniform")(x)
    
    x = Dense(50, activation='relu', kernel_initializer="uniform")(x)
    x = Dropout(0.35)(x)
    output_layer = Dense(1, activation='linear', kernel_initializer="uniform", name='main_output')(x)
    return keras.Model(inputs=input_layer, outputs=output_layer)

model_mcd = mcd()
model_mcd.summary()
model_mcd.compile(
            optimizer=keras.optimizers.Adam(learning_rate = LR_mcd),
            loss = 'mse',
            metrics=[
                # tf.keras.metrics.AUC(name='roc_auc', curve='ROC'),
                # tf.keras.metrics.AUC(name='pr_auc', curve='PR'),
                # tf.keras.metrics.Precision(name='precision'),
                # tf.keras.metrics.Recall(name='recall'),
                # tf.keras.metrics.BinaryAccuracy(name='accuracy')
                    ]
            )
hist_mcd = model_mcd.fit(xtrain, ytrain,
                     batch_size = BS_mcd,
                     epochs = EN_mcd,
                     validation_data = (xval_fnn, yval),
                     verbose = 1 )#, callbacks=[es]) 

loss_mcd.append(hist_mcd.history['loss'])
validation_loss_mcd.append([hist_mcd.history['val_loss']])
predictions_mcd = model_mcd.predict(xtest_fnn)
coef, p = spearmanr(np.squeeze(ytest), np.squeeze(predictions_mcd))
corr, p1 = pearsonr(np.squeeze(ytest), np.squeeze(predictions_mcd))
pearson_mcd.append(corr)
spearman_mcd.append(coef)

plt.figure()
plt.plot(hist_mcd.history['loss'], label='Train Loss')
plt.plot(hist_mcd.history['val_loss'], label='Validation Loss')
plt.title('MCD Training Curves')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

r2_mcd = r2_score(ytest, predictions_mcd)
mae_mcd = mean_absolute_error(ytest, predictions_mcd)
mape_mcd = np.mean(np.abs((ytest - predictions_mcd) / ytest)) * 100
rmse_mcd = np.sqrt(mean_squared_error(ytest, predictions_mcd))
mse_mcd = mean_squared_error(ytest, predictions_mcd)

print()
print(f"Pearson: {pearson_mcd}")
print(f"Spearman: {spearman_mcd}")
print(f"R²: {r2_mcd:.4f}")
print(f"MSE: {mse_mcd:.4f}")
print(f"MAE: {mae_mcd:.4f}")
print(f"RMSE: {rmse_mcd:.4f}")
print(f"MAPE: {mape_mcd:.2f}%")

#%%  noisy MLP pure 不可与mcd同时使用
def transformImages(
        xtrain, xtest,
        ytrain, ytest,
        imgrows, imgcols,
                    ):
    if tf.keras.backend.image_data_format() == 'channels_first':
        xtrain = xtrain.reshape(xtrain.shape[0], imgrows * imgcols)
        xtest = xtest.reshape(xtest.shape[0], imgrows * imgcols)
        input_shape = (1, imgrows, imgcols)
    else:
        xtrain = xtrain.reshape(xtrain.shape[0], imgrows * imgcols)
        xtest = xtest.reshape(xtest.shape[0], imgrows * imgcols)
        input_shape = (imgrows * imgcols)
    xtrain = xtrain.astype('float32')
    xtest = xtest.astype('float32')
    return xtrain, xtest, ytrain, ytest, input_shape
xtrain, xtest, ytrain, ytest, input_shape = transformImages(xtrain,xtest,ytrain,ytest,inputRow,inputCol)
# X2 = X.transpose(0,2,1)
# X = X2.reshape((len(X), inputRow*inputCol))
xtest_fnn = xtest

loss_mcd = []
validation_loss_mcd = []
pearson_mcd = []
spearman_mcd = []
# epoch number (EN); batch size (BS); learning rate (LR)
EN_mcd = 20
BS_mcd = 400
LR_mcd = 0.0001

def mcd():    
    input_layer = Input(shape=input_shape, name='main_input')
    x = Dense(10, activation='relu', kernel_initializer="uniform")(input_layer)#50 neuron
    # x = BatchNormalization()(x)
    x = Dropout(0.6)(x)
    # x = Dense(75, activation='relu', kernel_initializer="uniform")(x)
    # x = BatchNormalization()(x)
    x = Dense(10, activation='relu', kernel_initializer="uniform")(x)
    # x = BatchNormalization()(x)
    # x = Dropout(0.05)(x)
    # x = Dense(25, activation='relu', kernel_initializer="uniform")(x)
    
    # x = Dense(50, activation='relu', kernel_initializer="uniform")(x)
    x = Dropout(0.6)(x)
    output_layer = Dense(1, activation='linear', kernel_initializer="uniform", name='main_output')(x)
    return keras.Model(inputs=input_layer, outputs=output_layer)

model_mcd = mcd()
model_mcd.summary()
model_mcd.compile(
            optimizer=keras.optimizers.Adam(learning_rate = LR_mcd),
            loss = 'mse',
            metrics=[
                # tf.keras.metrics.AUC(name='roc_auc', curve='ROC'),
                # tf.keras.metrics.AUC(name='pr_auc', curve='PR'),
                # tf.keras.metrics.Precision(name='precision'),
                # tf.keras.metrics.Recall(name='recall'),
                # tf.keras.metrics.BinaryAccuracy(name='accuracy')
                    ]
            )
hist_mcd = model_mcd.fit(xtrain, ytrain,
                     batch_size = BS_mcd,
                     epochs = EN_mcd,
                     validation_data = (xval_fnn, yval),
                     verbose = 1 )#, callbacks=[es]) 

loss_mcd.append(hist_mcd.history['loss'])
validation_loss_mcd.append([hist_mcd.history['val_loss']])
predictions_mcd = model_mcd.predict(xtest_fnn)
coef, p = spearmanr(np.squeeze(ytest), np.squeeze(predictions_mcd))
corr, p1 = pearsonr(np.squeeze(ytest), np.squeeze(predictions_mcd))
pearson_mcd.append(corr)
spearman_mcd.append(coef)

plt.figure()
plt.plot(hist_mcd.history['loss'], label='Train Loss')
plt.plot(hist_mcd.history['val_loss'], label='Validation Loss')
plt.title('MCD Training Curves')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

r2_mcd = r2_score(ytest, predictions_mcd)
mae_mcd = mean_absolute_error(ytest, predictions_mcd)
mape_mcd = np.mean(np.abs((ytest - predictions_mcd) / ytest)) * 100
rmse_mcd = np.sqrt(mean_squared_error(ytest, predictions_mcd))
mse_mcd = mean_squared_error(ytest, predictions_mcd)

print()
print(f"Pearson: {pearson_mcd}")
print(f"Spearman: {spearman_mcd}")
print(f"R²: {r2_mcd:.4f}")
print(f"MSE: {mse_mcd:.4f}")
print(f"MAE: {mae_mcd:.4f}")
print(f"RMSE: {rmse_mcd:.4f}")
print(f"MAPE: {mape_mcd:.2f}%")


#%% print all reg metrics 
print()
print(f"FNN Pearson: {pearson_fnn}")
print(f"FNN Spearman: {spearman_fnn}")
print(f"FNN R²: {r2_fnn:.4f}")
print(f"FNN MSE: {mse_fnn:.4f}")
print(f"FNN MAE: {mae_fnn:.4f}")
print(f"FNN RMSE: {rmse_fnn:.4f}")
print(f"FNN MAPE: {mape_fnn:.2f}%")

print()
print(f"CNN Pearson: {pearson_cnn}")
print(f"CNN Spearman: {spearman_cnn}")
print(f"CNN R²: {r2_cnn:.4f}")
print(f"CNN MSE: {mse_cnn:.4f}")
print(f"CNN MAE: {mae_cnn:.4f}")
print(f"CNN RMSE: {rmse_cnn:.4f}")
print(f"CNN MAPE: {mape_cnn:.2f}%")

print()
print(f"RNN Pearson: {pearson_rnn}")
print(f"RNN Spearman: {spearman_rnn}")
print(f"RNN R²: {r2_rnn:.4f}")
print(f"RNN MSE: {mse_rnn:.4f}")
print(f"RNN MAE: {mae_rnn:.4f}")
print(f"RNN RMSE: {rmse_rnn:.4f}")
print(f"RNN MAPE: {mape_rnn:.2f}%")


#%% equally ensemble version 2
import numpy as np
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from scipy.stats import spearmanr, pearsonr

def ensemble_evaluate_equal_weight_general(ytest, *predictions):
    """
    Equally ensembles N model outputs (2 <= N <= 10) and evaluates performance.

    Parameters:
        ytest (np.ndarray): Ground truth values.
        *predictions: Variable number of prediction arrays (from 2 to 10 models).

    Returns:
        dict: A dictionary containing ensemble performance metrics.
    """

    # Convert ytest and predictions to arrays
    ytest = np.array(ytest)
    preds_list = [np.array(p) for p in predictions]

    # === Step 1: Validity Checks ===
    n_models = len(preds_list)
    if not (2 <= n_models <= 10):
        raise ValueError(f"Number of models must be between 2 and 10, got {n_models}.")

    for i, p in enumerate(preds_list):
        if p.shape != ytest.shape:
            raise ValueError(f"Prediction at index {i} does not match shape of ytest. "
                             f"Got {p.shape}, expected {ytest.shape}.")

    # === Step 2: Compute average ensemble ===
    stacked_preds = np.stack(preds_list, axis=0)  # shape: (n_models, ...)
    final_predictions = np.mean(stacked_preds, axis=0)

    # === Step 3: Compute metrics ===
    r2 = r2_score(ytest, final_predictions)
    mae = mean_absolute_error(ytest, final_predictions)
    mse = mean_squared_error(ytest, final_predictions)
    rmse = np.sqrt(mse)
    # mape = np.mean(np.abs((ytest - final_predictions) / ytest)) * 100

    ytest_flat = np.squeeze(ytest)
    preds_flat = np.squeeze(final_predictions)

    spearman_corr, _ = spearmanr(ytest_flat, preds_flat)
    pearson_corr, _ = pearsonr(ytest_flat, preds_flat)

    return {
        "n_models": n_models,        
        "pearson_corr": pearson_corr,
        "spearman_corr": spearman_corr,
        "r2": r2,
        "mse": mse,
        "mae": mae,
        "rmse": rmse,
        # "mape": mape  
    }

# results_equal = ensemble_evaluate_equal_weight_general(ytest, predictions_fnn, predictions_cnn, predictions_rnn)
# print(results_equal)



# ensemble_evaluate_equal_weight_general(ytest, predictions_fnn, predictions_cnn, predictions_rnn, predictions_attention)





# Validation predictions use dedicated inputs and names; test predictions are not overwritten.
predictions_fnn_val = model_fnn.predict(xval_fnn)
predictions_cnn_val = model_cnn.predict(xval_cnn)
predictions_rnn_val = model_rnn.predict(xval_rnn)
predictions_attention_val = model_attention.predict(xval_rnn)
ensemble_evaluate_equal_weight_general(
    yval,
    predictions_fnn_val,
    predictions_cnn_val,
    predictions_rnn_val,
    predictions_attention_val,
)

#%% ablation study: subnetwork combinations (equal-weight ensemble), DE Figure
# model_mcd here is the noisy MLP (the "noisy MLP pure" cell overwrites the MC-dropout one)
ablation_configs = {
    'CRISPR-DE (full ensemble)': ['MLP', 'CNN', 'LSTM', 'Attention'],
    'NO-MLP':                    ['CNN', 'LSTM', 'Attention'],
    'NO-CNN':                    ['MLP', 'LSTM', 'Attention'],
    'NO-LSTM':                   ['MLP', 'CNN', 'Attention'],
    'NO-Attention':              ['MLP', 'CNN', 'LSTM'],
    'Add noisy MLP':             ['MLP', 'CNN', 'LSTM', 'Attention', 'Noisy MLP'],
}

def ablation_table(y_true, preds_by_model):
    rows = []
    for config_name, members in ablation_configs.items():
        res = ensemble_evaluate_equal_weight_general(y_true, *[preds_by_model[m] for m in members])
        rows.append({
            'Configuration': config_name,
            'Pearson': res['pearson_corr'],
            'Spearman': res['spearman_corr'],
            'MAE': res['mae'],
            'MSE': res['mse'],
            'RMSE': res['rmse'],
        })
    return pd.DataFrame(rows)

preds_val = {
    'MLP': predictions_fnn_val,
    'CNN': predictions_cnn_val,
    'LSTM': predictions_rnn_val,
    'Attention': predictions_attention_val,
    'Noisy MLP': model_mcd.predict(xval_fnn),
}
preds_test = {
    'MLP': model_fnn.predict(xtest_fnn),
    'CNN': model_cnn.predict(xtest_cnn),
    'LSTM': model_rnn.predict(xtest_rnn),
    'Attention': model_attention.predict(xtest_rnn),
    'Noisy MLP': model_mcd.predict(xtest_fnn),
}

ablation_val = ablation_table(yval, preds_val)
ablation_test = ablation_table(ytest, preds_test)

pd.set_option('display.width', 200)
print('\n=== Ablation (validation set, n=%d) ===' % len(yval))
print(ablation_val.to_string(index=False, float_format='%.5f'))
print('\n=== Ablation (test set, n=%d) ===' % len(ytest))
print(ablation_test.to_string(index=False, float_format='%.5f'))

ablation_val.to_csv(os.path.join(RESULT_DIR, 'ablation_validation.csv'), index=False)
ablation_test.to_csv(os.path.join(RESULT_DIR, 'ablation_test.csv'), index=False)
print('Saved ablation results to', RESULT_DIR)
