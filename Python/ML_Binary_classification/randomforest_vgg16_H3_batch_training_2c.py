# Import necessary modules
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score,recall_score,f1_score,confusion_matrix,classification_report
from sklearn.metrics import roc_curve,auc,roc_auc_score
from sklearn.model_selection import train_test_split
from scipy.signal import savgol_filter

import tensorflow as tf

from tensorflow.keras.applications import VGG16
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import KFold
from PIL import Image

# Define constants for image folder and label paths, and image size
IMAGE_FOLDER_PATH = "C:/Users/truon011/Image_data_9_9_24/RGB"
LABELS_CSV_PATH = "C:/Users/truon011/labels_2c.csv"
IMAGE_SIZE = (224, 224)

# Initialize scalers and encoders
scaler = StandardScaler()
le = LabelEncoder()

# Restructured code for Random Forest classifier with ResNet50
   
def batch_image_loader(folder_path, csv_path, batch_size=500):
    """
    A generator that loads images and labels in batches from the given folder and CSV file, 
    ensuring an equal number of samples from both classes ("A" and "B") in each batch.
    """
    # Read the CSV file and filter for classes "A" and "B"
    df_labels = pd.read_csv(csv_path, index_col="sequence_ID")
    df_labels_filtered = df_labels[df_labels["class_label"].isin(["A", "B"])]
    df_labels_filtered.index = df_labels_filtered.index.astype(str)

    # Calculate half of the batch size to ensure equal class distribution
    half_batch_size = batch_size // 2

    images_class_a = []
    labels_class_a = []
    images_class_b = []
    labels_class_b = []

    # Iterate over the files in the image folder
    for filename in os.listdir(folder_path):
        if filename.endswith(".png"):
            # Extract the sequence ID from the filename
            sequence_id = os.path.splitext(filename)[0].lstrip("0")
            sequence_id = str(sequence_id).zfill(len(df_labels_filtered.index[0]))

            # Check if the sequence ID exists in the filtered labels
            if sequence_id in df_labels_filtered.index:
                label = df_labels_filtered.loc[sequence_id, "class_label"]
                img_path = os.path.join(folder_path, filename)
                
                # Load and preprocess the image
                img = Image.open(img_path)
                img = img.convert("RGB")
                img = img.resize(IMAGE_SIZE, resample=Image.LANCZOS)
                img_array = np.array(img)

                # Append the image to the respective class list
                if label == "A":
                    images_class_a.append(img_array)
                    labels_class_a.append(label)
                elif label == "B":
                    images_class_b.append(img_array)
                    labels_class_b.append(label)

                # If both class lists have reached half the batch size, combine and yield them
                if len(images_class_a) >= half_batch_size and len(images_class_b) >= half_batch_size:
                    # Combine class "A" and class "B" images and labels
                    images = images_class_a[:half_batch_size] + images_class_b[:half_batch_size]
                    labels = labels_class_a[:half_batch_size] + labels_class_b[:half_batch_size]

                    # Encode the labels
                    labels_encoded = le.fit_transform(labels)

                    # Yield the batch of images and labels
                    yield np.array(images), np.array(labels_encoded)

                    # Remove the yielded images and labels from the lists
                    images_class_a = images_class_a[half_batch_size:]
                    labels_class_a = labels_class_a[half_batch_size:]
                    images_class_b = images_class_b[half_batch_size:]
                    labels_class_b = labels_class_b[half_batch_size:]

    # Yield any remaining images if they form a complete batch
    while len(images_class_a) >= half_batch_size and len(images_class_b) >= half_batch_size:
        images = images_class_a[:half_batch_size] + images_class_b[:half_batch_size]
        labels = labels_class_a[:half_batch_size] + labels_class_b[:half_batch_size]
        labels_encoded = le.fit_transform(labels)
        yield np.array(images), np.array(labels_encoded)
        images_class_a = images_class_a[half_batch_size:]
        labels_class_a = labels_class_a[half_batch_size:]
        images_class_b = images_class_b[half_batch_size:]
        labels_class_b = labels_class_b[half_batch_size:]

def normalization(data):
    data_normalized= (data - np.min(data)) / (np.max(data) - np.min(data))
    return data_normalized

def smoothing(data):
    data_sm=savgol_filter(data, window_length=3, polyorder=2, deriv=0,axis=1)
    return data_sm

def preprocess_data(data, labels, test_size=0.2):
    """
    Splits the data into training,and test sets.
    Returns the split data and labels.
    """
    data_sm=smoothing(data)
    data_norm=normalization(data_sm)
    x_train, x_test, y_train, y_test = train_test_split(data_norm, labels, test_size=test_size, random_state=42)
    return x_train, x_test, y_train, y_test

def extract_features(VGG16_model, data):
    """
    Extracts features from the given data using the ResNet50 model.
    Returns the extracted features.
    """
    features = VGG16_model.predict(data)
    flattened_features = features.reshape((features.shape[0], -1))
    return flattened_features

def evaluate_model(model, data, labels):
    y_pred = model.predict(data)
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

def evaluate_model_mean_std(models, data, labels):
    accuracy_scores = []
    precision_scores=[]
    recall_scores=[]
    f1_scores=[]
    for model in models:
        results = evaluate_model(model, data, labels)
        accuracy_scores.append(results['accuracy'])
        precision_scores.append(results['precision'])
        recall_scores.append(results['recall'])
        f1_scores.append(results['f1'])
        
    mean_accuracy = np.mean(accuracy_scores)
    std_accuracy = np.std(accuracy_scores)
    mean_precision=np.mean(precision_scores)
    mean_recall=np.mean(recall_scores)
    mean_f1=np.mean(f1_scores)
    
    return mean_accuracy, std_accuracy,mean_precision,mean_recall,mean_f1

def accumulate_predictions_labels(model, batch_data, batch_labels):
    """
    Predicts and accumulates predictions and true labels for each batch.
    """
    y_pred = model.predict(batch_data)  
    return y_pred, batch_labels

def create_rf_model(data, labels, n_estimators=300):
    """
    Creates and trains a Random Forest model using 3 K-Fold cross-validation.
    """
    data = data.reshape((data.shape[0], -1))  # Flatten the data 

    kfold = KFold(n_splits=3, shuffle=True, random_state=42)  # Internal validation with Kfold

    fold_results = []

    for train_index, val_index in kfold.split(data, labels):
        x_train, x_val = data[train_index], data[val_index]
        y_train, y_val = labels[train_index], labels[val_index]

        # Initialize the model for each fold
        model = RandomForestClassifier(max_depth=20, n_estimators=n_estimators, min_samples_leaf=4,
                                       min_samples_split=10, criterion="entropy", max_features='sqrt', random_state=42)

        model.fit(x_train, y_train)

        # Evaluate the model on the validation set
        val_results = evaluate_model(model, x_val, y_val)
        fold_results.append(val_results)
        
        # Print the validation results for the current fold
        print("Validation Results for current fold:")
        for key, value in val_results.items():
            if key != 'class_metrics':
                print(f"{key.capitalize()}: {value}")
            else:
                print(f"{key.capitalize()}:")
                for class_key, class_value in value.items():
                    print(f"  {class_key}: {class_value}")
        print("=" * 30)

    return model

def plot_roc_curve(y_true, y_probs):
        fpr, tpr, _ = roc_curve(y_true, y_probs)
        roc_auc = roc_auc_score(y_true, y_probs)
        plt.figure()
        plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Specificity')
        plt.ylabel('Sensitivity')
        plt.title('ROC Curve')
        plt.legend(loc="lower right")
        plt.show()
        print('/n' + '='*30 + '/n')
        
def create_and_evaluate_models_with_batches(folder_path, csv_path, VGG16_model, batch_size=500, epochs=5):
    """
    Trains and evaluates the model using batch training. 
    """
    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}")
        batch_loader = batch_image_loader(folder_path, csv_path, batch_size)
        
        # Initialize accumulators for predictions and true labels
        all_y_pred = []
        all_y_true = []
        
        for batch_data, batch_labels in batch_loader:
            x_train, x_test, y_train, y_test = preprocess_data(batch_data, batch_labels)
            
            x_train_features = extract_features(VGG16_model, x_train)
            x_test_features = extract_features(VGG16_model, x_test)

            rf_model = create_rf_model(x_train_features, y_train)
            
            # Evaluate model on test data and accumulate predictions
            y_pred_batch, y_true_batch = accumulate_predictions_labels(rf_model, x_test_features, y_test)
            all_y_pred.extend(y_pred_batch)  
            all_y_true.extend(y_true_batch)  
            
            train_results = evaluate_model(rf_model, x_train_features, y_train)
            test_results = evaluate_model(rf_model, x_test_features, y_test)
            
            # Get probabilities and handle cases with only one class
            y_probs = rf_model.predict_proba(x_test_features)
            if y_probs.shape[1] == 2:
               y_probs = y_probs[:, 1]  # Probabilities for class 1
            else:
               y_probs = y_probs[:, 0]  # Only class 0, use its probability

            plot_roc_curve(y_test, y_probs)
            
            print("Training Results:")
            for key, value in train_results.items():
                print(f"{key.capitalize()}: {value}")

            print("Test Results:")
            for key, value in test_results.items():
                print(f"{key.capitalize()}: {value}")
        
        # Convert accumulated predictions and true labels to numpy arrays
        all_y_pred = np.array(all_y_pred)
        all_y_true = np.array(all_y_true)
        
        # Compute confusion matrix after all batches
        confusion_mt = confusion_matrix(all_y_true, all_y_pred)
        
        # Print confusion matrix and classification report
        print("Confusion Matrix for the entire epoch:")
        print(confusion_mt)

        print("Classification Report:")
        print(classification_report(all_y_true, all_y_pred))
        
        mean_accuracy, std_accuracy, mean_precision, mean_recall, mean_f1 = evaluate_model_mean_std([rf_model], x_test_features, y_test)
        print(f"Epoch {epoch + 1} Mean Accuracy: {mean_accuracy}")
        print(f"Epoch {epoch + 1} Standard Accuracy: {std_accuracy}")
        print(f"Epoch {epoch + 1} Mean Precision: {mean_precision}")
        print(f"Epoch {epoch + 1} Mean Recall: {mean_recall}")
        print(f"Epoch {epoch + 1} Mean F1: {mean_f1}")

if __name__ == "__main__":
    data_folder_path = IMAGE_FOLDER_PATH
    csv_file_path = LABELS_CSV_PATH
    
    # Initialize VGG16 model
    VGG16_model = VGG16(input_shape=(*IMAGE_SIZE, 3), weights='imagenet', include_top=False)
    VGG16_model.trainable = False

    # Train and evaluate using batch processing
    create_and_evaluate_models_with_batches(data_folder_path, csv_file_path, VGG16_model, batch_size=500, epochs=5)


