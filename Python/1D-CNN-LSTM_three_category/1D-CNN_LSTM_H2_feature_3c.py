import os
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder,label_binarize
from sklearn.metrics import accuracy_score, precision_score,recall_score,f1_score,confusion_matrix,classification_report
from sklearn.metrics import roc_curve,auc,roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold

import tensorflow as tf
from tensorflow import keras 
from tensorflow.keras.models import Sequential,load_model
from tensorflow.keras.layers import Conv1D,LSTM, Reshape
from tensorflow.keras.layers import Dense, Dropout, Flatten,MaxPooling1D,Activation
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping,Callback
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.compat.v1.keras.backend import set_session
from PIL import Image

# Call the function transform labels
le=LabelEncoder()

# Import data and label paths
data_csv_paths = [
    "C:/Users/truon011/OneDrive - Wageningen University & Research/Wageningen UR _ Research_Work/Project_July_Nov_24/Raw_data_29_7_24/Data_random_model_3classes/Three_compartments/simu_Ba_i_3c_T.csv",
    "C:/Users/truon011/OneDrive - Wageningen University & Research/Wageningen UR _ Research_Work/Project_July_Nov_24/Raw_data_29_7_24/Data_random_model_3classes/Three_compartments/simu_Ba_r_3c_T.csv",
    "C:/Users/truon011/OneDrive - Wageningen University & Research/Wageningen UR _ Research_Work/Project_July_Nov_24/Raw_data_29_7_24/Data_random_model_3classes/Three_compartments/simu_Ma_i_3c_T.csv"
]
label_csv_path = "C:/Users/truon011/OneDrive - Wageningen University & Research/Wageningen UR _ Research_Work/Project_July_Nov_24/Raw_data_29_7_24/Data_random_model_3classes/labels_3c.csv"


def load_csv_files(csv_paths):
    """
    Load multiple CSV files and return a list of DataFrames.
    """
    dataframes = [pd.read_csv(csv_path, header=0) for csv_path in csv_paths]
    return dataframes
    
def load_labels_from_csv(csv_path,num_labels=10368):
    df_labels = pd.read_csv(csv_path, index_col="sequence_ID")
    labels = df_labels["class_label"].values[:num_labels]
    labels=le.fit_transform(labels)
    return labels  

def merge_time_series_with_labels(time_series, labels):
    """
    Merge corresponding rows from multiple time series dataframes.
    Each sample will consist of one row from each dataframe.
    Returns merged data and corresponding labels.
    """
    merged_data = []
    merged_labels = []

    # Number of rows in the dataframes
    num_rows = len(labels)

    # Iterate over each row index
    for i in range(num_rows):
        # Combine rows from each dataframe (A, B, C) for the i-th row
        combined_row = np.concatenate([time_series[j].iloc[i].values for j in range(len(time_series))])
        
        # Append the combined row and the corresponding label
        merged_data.append(combined_row)
        merged_labels.append(labels[i])
    
    # Convert lists to numpy arrays
    return np.array(merged_data), np.array(merged_labels)
 
def normalization(data):
    data_normalized= (data - np.min(data)) / (np.max(data) - np.min(data))
    return data_normalized

def preprocess_data(data, labels, test_size=0.2):
    """
    Splits the data into training, validation, and test sets.
    Returns the split data and labels.
    """
    data_norm=normalization(data)
    x_train, x_test, y_train, y_test = train_test_split(data_norm, labels, test_size=test_size, random_state=42)
    return x_train, x_test, y_train, y_test

def create_conv_layer(model, filters, kernel_size, dropout_rate):
    model.add(Conv1D(filters=filters, kernel_size=kernel_size, strides=2, padding='valid', activation='relu'))
    model.add(MaxPooling1D(pool_size=2, strides=2, padding='valid'))
    model.add(Dropout(dropout_rate))
    
def create_lstm_layer(model, filters, kernel_size, dropout_rate):
    model.add(LSTM(filters=filters, kernel_size=kernel_size, strides=2, padding='valid', activation='relu'))
    model.add(MaxPooling1D(pool_size=2, strides=2, padding='valid'))
    model.add(Dropout(dropout_rate))
    
def create_cnn_lstm_model(data, labels, num_classes, learning_rate, batch_size, epochs):
    model = Sequential()
    
    # Convolutional Layers
    model.add(Conv1D(64, 3, strides=2, padding='valid', activation='relu', input_shape=(data.shape[1], 1)))
    model.add(MaxPooling1D(pool_size=2, strides=2, padding='valid'))
    model.add(Dropout(0.1))
    
    model.add(Conv1D(128, 3, strides=2, padding='valid', activation='relu'))
    model.add(MaxPooling1D(pool_size=2, strides=2, padding='valid'))
    model.add(Dropout(0.1))
    
    model.add(Conv1D(256, 3, strides=2, padding='valid', activation='relu'))
    model.add(MaxPooling1D(pool_size=2, strides=2, padding='valid'))
    model.add(Dropout(0.1))
    
    # LSTM Layer
    # Reshape the output to (timesteps, features) for LSTM
    # Reshape CNN output for LSTM input
    # Calculate the shape of the output after the CNN layers
    cnn_output_shape = model.output_shape
    cnn_output_length = cnn_output_shape[1]  # Length of the output sequence after CNN
    num_cnn_features = cnn_output_shape[2]  # Number of features from the last Conv1D

    # Reshape CNN output for LSTM
    timesteps = cnn_output_length
    features_per_timestep = num_cnn_features
    
    # Reshape CNN output for LSTM
    model.add(Reshape((timesteps, features_per_timestep)))
  
    model.add(LSTM(128, return_sequences=False))  # 128 units in LSTM layer
    
    # Dense Layers
    model.add(Dense(units=512, activation='relu'))
    model.add(Dense(units=num_classes, activation='softmax'))
    
    # Compile the Model
    Adam = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(loss='categorical_crossentropy', optimizer=Adam, metrics=['categorical_accuracy'])
    model.summary()
  
    return model

def evaluate_model(model, data, labels):
    y_pred_ = model.predict(data)
    y_pred = np.argmax(y_pred_, axis=1)
    labels = np.argmax(labels, axis=1)
    
    precision = precision_score(labels, y_pred, average='weighted')
    recall = recall_score(labels, y_pred, average='weighted')
    f1 = f1_score(labels, y_pred, average='weighted')
    acc = accuracy_score(labels, y_pred)
    number_acc = accuracy_score(labels, y_pred, normalize=False)
    confusion_mt = confusion_matrix(labels, y_pred)
    report = classification_report(labels, y_pred, output_dict=True)
    
    class_metrics = {
        'accuracy': acc,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'accuracy_count': number_acc,
        'confusion_matrix': confusion_mt,
        'class_metrics': report
    }
    
    return class_metrics

def save_data_as_csv(x_train, x_test, y_train, y_test):
    # Convert data to pandas DataFrames
    x_train_df = pd.DataFrame(x_train)
    x_test_df = pd.DataFrame(x_test)
    y_train_df = pd.DataFrame(y_train)
    y_test_df = pd.DataFrame(y_test)

    # Save the data and labels as CSV
    x_train_df.to_csv('x_train.csv', index=False)
    x_test_df.to_csv('x_test.csv', index=False)
    y_train_df.to_csv('y_train.csv', index=False)
    y_test_df.to_csv('y_test.csv', index=False)
    
# Create and evaluate models using k-fold cross-validation
def create_and_evaluate_models(data, labels, num_classes=3, learning_rate=0.0001, batch_size=50, epochs=100, n_splits=5):    
    # Split data into training and independent test set
    x_train, x_test, y_train, y_test = preprocess_data(data, labels, test_size=0.2)
    # Save training and test data and labels to CSV
    save_data_as_csv(x_train, x_test, y_train, y_test)
    
    y_train_one_hot=keras.utils.to_categorical(y_train, num_classes=num_classes)
    y_test_one_hot = keras.utils.to_categorical(y_test, num_classes=num_classes)
    
    kfold = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    fold_results = []

    for train_index, val_index in kfold.split(x_train, y_train):
        x_train_fold, x_val_fold = x_train[train_index], x_train[val_index]
        y_train_fold, y_val_fold = y_train[train_index], y_train[val_index]

        y_train_fold_one_hot = keras.utils.to_categorical(y_train_fold, num_classes=num_classes)
        y_val_fold_one_hot = keras.utils.to_categorical(y_val_fold, num_classes=num_classes)

        model = create_cnn_lstm_model(x_train_fold, y_train_one_hot, num_classes=num_classes, learning_rate=learning_rate,batch_size=batch_size, epochs=epochs)

        callbacks = [
            ModelCheckpoint(filepath='model_best_cnn_lstm_3c.h5', monitor='val_loss', save_best_only=True, verbose=0),
            EarlyStopping(monitor='val_loss', patience=10)
        ]

        history = model.fit(x_train_fold, y_train_fold_one_hot, epochs=epochs, batch_size=batch_size, validation_data=(x_val_fold, y_val_fold_one_hot), verbose=1, callbacks=callbacks)

        # Plot training history
        plt.plot(history.history['loss'], label='Training loss')
        plt.plot(history.history['val_loss'], label='Validation loss')
        plt.legend()
        plt.xlabel('Epochs')
        plt.ylabel('Loss value')
        plt.show()

        val_results = evaluate_model(model, x_val_fold, y_val_fold_one_hot)
        fold_results.append(val_results)

        print("Validation Results for current fold:")
        for key, value in val_results.items():
            if key != 'class_metrics':
                print(f"{key.capitalize()}: {value}")
            else:
                print(f"{key.capitalize()}:")
                for class_key, class_value in value.items():
                    print(f"  {class_key}: {class_value}")
        print("=" * 30)
        
        # Clear session to free up memory
        keras.backend.clear_session()
        
    # Evaluate on the independent test set
    print("Evaluating on the independent test set:")
    train_results= evaluate_model(model,x_train,y_train_one_hot)
    test_results = evaluate_model(model, x_test, y_test_one_hot)

    return train_results, test_results
if __name__ == "__main__":
     # Load data and labels
    dataframes = load_csv_files(data_csv_paths)
    labels = load_labels_from_csv(label_csv_path)
    labels = labels[0:10368]
    # Prepare data and group labels
    merged_data, merged_labels = merge_time_series_with_labels(dataframes, labels)
   
    train_results, test_results = create_and_evaluate_models(merged_data, merged_labels)
    print(f"Final train results: {train_results}")
    print(f"Final test results: {test_results}")

