import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import pandas as pd
import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, concatenate
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import seaborn as sns

def build_train_test_cnn_model(X_train, y_train, X_val, y_val, X_test, y_test, filters=32, kernel_size=3, pool_size=2,
                           dense_units=128, dropout_rate=0.5, epochs=10, batch_size=32, learning_rate=0.001,
                           use_early_stopping=True, patience=3, num_conv_layers=2):

    physical_devices = tf.config.list_physical_devices('GPU')
    if physical_devices:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
        print("GPU is available.")
    else:
        print("No GPU detected. Running on CPU.")

    model = Sequential()

    for i in range(num_conv_layers):
        model.add(Conv1D(filters=filters*(2**i), kernel_size=kernel_size, activation='relu',
                         input_shape=(X_train.shape[1], 1)))
        model.add(MaxPooling1D(pool_size=pool_size))

    model.add(Flatten())
    model.add(Dense(dense_units, activation='relu'))
    model.add(Dropout(dropout_rate))
    model.add(Dense(1, activation='sigmoid'))

    optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

    callbacks = []
    if use_early_stopping:
        early_stopping = EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True)
        callbacks.append(early_stopping)

    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_val, y_val),
                        callbacks=callbacks, verbose=1)

    y_pred = model.predict(X_test)
    y_pred_binary = (y_pred > 0.5).astype(int)


    test_accuracy = accuracy_score(y_test, y_pred_binary)
    precision = precision_score(y_test, y_pred_binary)
    recall = recall_score(y_test, y_pred_binary)
    f1 = f1_score(y_test, y_pred_binary)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred_binary).ravel()
    specificity = tn / (tn + fp)
    cm = confusion_matrix(y_test, y_pred_binary)

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['0', '1'], yticklabels=['0', '1'])
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted label')
    plt.ylabel('True label')

    plt.show()

    _, test_accuracy = model.evaluate(X_test, y_test, verbose=0)

    return test_accuracy, precision, recall, f1, specificity

train_data = pd.read_csv('./MyDrive/MyDrive/metriseis_data/train.csv')
X_train = train_data.iloc[:-1,:-1]
Y_train = train_data.iloc[:-1,-1]

test_data = pd.read_csv('./MyDrive/MyDrive/metriseis_data/test.csv')
X_test = test_data.iloc[:-1,:-1]
Y_test = test_data.iloc[:-1,-1]

validate_data = pd.read_csv('./MyDrive/MyDrive/metriseis_data/validate.csv')
X_val = validate_data.iloc[:-1,:-1]
Y_val = validate_data.iloc[:-1,-1]

epoch = 20
hyperparameters_to_test = [

    # Different Filters
    {'filters': 32, 'kernel_size': 3, 'pool_size': 2, 'dense_units': 128, 'dropout_rate': 0.5, 'epochs': epoch, 'batch_size': 32, 'learning_rate': 0.001, 'use_early_stopping': True, 'patience': 3, 'num_conv_layers': 4},
    {'filters': 64, 'kernel_size': 3, 'pool_size': 2, 'dense_units': 128, 'dropout_rate': 0.5, 'epochs': epoch, 'batch_size': 32, 'learning_rate': 0.001, 'use_early_stopping': True, 'patience': 3, 'num_conv_layers': 4},
    {'filters': 16, 'kernel_size': 3, 'pool_size': 2, 'dense_units': 128, 'dropout_rate': 0.5, 'epochs': epoch, 'batch_size': 32, 'learning_rate': 0.001, 'use_early_stopping': True, 'patience': 3, 'num_conv_layers': 4},
    {'filters': 128, 'kernel_size': 3, 'pool_size': 2, 'dense_units': 128, 'dropout_rate': 0.5, 'epochs': epoch, 'batch_size': 32, 'learning_rate': 0.001, 'use_early_stopping': True, 'patience': 3, 'num_conv_layers': 4},

    #Different Kernel Sizes
    {'filters': 64, 'kernel_size': 4, 'pool_size': 2, 'dense_units': 128, 'dropout_rate': 0.5, 'epochs': epoch, 'batch_size': 32, 'learning_rate': 0.001, 'use_early_stopping': True, 'patience': 3, 'num_conv_layers': 4},
    {'filters': 64, 'kernel_size': 5, 'pool_size': 2, 'dense_units': 128, 'dropout_rate': 0.5, 'epochs': epoch, 'batch_size': 32, 'learning_rate': 0.001, 'use_early_stopping': True, 'patience': 3, 'num_conv_layers': 4},
    {'filters': 64, 'kernel_size': 6, 'pool_size': 2, 'dense_units': 128, 'dropout_rate': 0.5, 'epochs': epoch, 'batch_size': 32, 'learning_rate': 0.001, 'use_early_stopping': True, 'patience': 3, 'num_conv_layers': 4},
    {'filters': 64, 'kernel_size': 7, 'pool_size': 2, 'dense_units': 128, 'dropout_rate': 0.5, 'epochs': epoch, 'batch_size': 32, 'learning_rate': 0.001, 'use_early_stopping': True, 'patience': 3, 'num_conv_layers': 4},

    #Different Dense Units
    {'filters': 64, 'kernel_size': 3, 'pool_size': 2, 'dense_units': 128, 'dropout_rate': 0.5, 'epochs': epoch, 'batch_size': 32, 'learning_rate': 0.001, 'use_early_stopping': True, 'patience': 3, 'num_conv_layers': 4},
    {'filters': 64, 'kernel_size': 3, 'pool_size': 2, 'dense_units': 256, 'dropout_rate': 0.5, 'epochs': epoch, 'batch_size': 32, 'learning_rate': 0.001, 'use_early_stopping': True, 'patience': 3, 'num_conv_layers': 4},
    {'filters': 64, 'kernel_size': 3, 'pool_size': 2, 'dense_units': 512, 'dropout_rate': 0.5, 'epochs': epoch, 'batch_size': 32, 'learning_rate': 0.001, 'use_early_stopping': True, 'patience': 3, 'num_conv_layers': 4},
    {'filters': 64, 'kernel_size': 3, 'pool_size': 2, 'dense_units': 1024, 'dropout_rate': 0.5, 'epochs': epoch, 'batch_size': 32, 'learning_rate': 0.001, 'use_early_stopping': True, 'patience': 3, 'num_conv_layers': 4},

    #Different Dropout
    {'filters': 64, 'kernel_size': 3, 'pool_size': 2, 'dense_units': 128, 'dropout_rate': 0.4, 'epochs': epoch, 'batch_size': 32, 'learning_rate': 0.001, 'use_early_stopping': True, 'patience': 3, 'num_conv_layers': 4},
    {'filters': 64, 'kernel_size': 3, 'pool_size': 2, 'dense_units': 128, 'dropout_rate': 0.3, 'epochs': epoch, 'batch_size': 32, 'learning_rate': 0.001, 'use_early_stopping': True, 'patience': 3, 'num_conv_layers': 4},
    {'filters': 64, 'kernel_size': 3, 'pool_size': 2, 'dense_units': 128, 'dropout_rate': 0.2, 'epochs': epoch, 'batch_size': 32, 'learning_rate': 0.001, 'use_early_stopping': True, 'patience': 3, 'num_conv_layers': 4},
    {'filters': 64, 'kernel_size': 3, 'pool_size': 2, 'dense_units': 128, 'dropout_rate': 0.1, 'epochs': epoch, 'batch_size': 32, 'learning_rate': 0.001, 'use_early_stopping': True, 'patience': 3, 'num_conv_layers': 4},

    #Different Batch Size
    {'filters': 64, 'kernel_size': 3, 'pool_size': 2, 'dense_units': 128, 'dropout_rate': 0.5, 'epochs': epoch, 'batch_size': 64, 'learning_rate': 0.001, 'use_early_stopping': True, 'patience': 3, 'num_conv_layers': 4},
    {'filters': 64, 'kernel_size': 3, 'pool_size': 2, 'dense_units': 128, 'dropout_rate': 0.5, 'epochs': epoch, 'batch_size': 128, 'learning_rate': 0.001, 'use_early_stopping': True, 'patience': 3, 'num_conv_layers': 4},
    {'filters': 64, 'kernel_size': 3, 'pool_size': 2, 'dense_units': 128, 'dropout_rate': 0.5, 'epochs': epoch, 'batch_size': 256, 'learning_rate': 0.001, 'use_early_stopping': True, 'patience': 3, 'num_conv_layers': 4},
    {'filters': 64, 'kernel_size': 3, 'pool_size': 2, 'dense_units': 128, 'dropout_rate': 0.5, 'epochs': epoch, 'batch_size': 512, 'learning_rate': 0.001, 'use_early_stopping': True, 'patience': 3, 'num_conv_layers': 4}

    ]

cnn_results = []

for params in hyperparameters_to_test:
    test_acc, precision, recall, f1, specificity = build_train_test_cnn_model(X_train, Y_train, X_val, Y_val, X_test, Y_test, **params)
    cnn_results.append((params, test_acc, precision, recall, f1, specificity))

file_path = './MyDrive/MyDrive/metriseis_data/cnn_results.csv'

with open(file_path, 'w') as file:
    file.write("Parameters,Accuracy,Precision,Recall,F1,Specificity\n")

    for params, result, precision, recall, f1, specificity in cnn_results:
        parameters_str = '_'.join(f"{key}_{value}" for key, value in params.items())
        file.write(f"{parameters_str},{result},{precision},{recall},{f1},{specificity}\n")
