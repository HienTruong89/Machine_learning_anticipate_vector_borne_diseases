import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import pdist, squareform
from PIL import Image
import matplotlib.pyplot as plt
import os

import os
os.chdir('C:/Users/truon011/OneDrive - Wageningen University & Research/Wageningen UR _ Research_Work/Project_July_Nov_24/Raw_data_10_9_24/Three_compartments')
directory='colorimage'

class TimeSeriesAnalyzer:
    def __init__(self):
        self.scaler = StandardScaler()

    def fit(self, feature_vector):
        """
        Fits the scaler on the feature vector.        
        Args:
            feature_vector (np.array): The feature vector to fit the scaler on.
        """
        self.scaler.fit(feature_vector)

    def transform(self, feature_vector):
        """
        Transforms the feature vector using the fitted scaler.        
        Args:
            feature_vector (np.array): The feature vector to transform.
        Returns:
            np.array: The transformed feature vector.
        """
        return self.scaler.transform(feature_vector)

    def fit_transform(self, feature_vector):
        """
        Fits the scaler on the feature vector and then transforms it..
        Returns:
            np.array: The fitted and transformed feature vector.
        """
        return self.scaler.fit_transform(feature_vector)

    def calculate_distance_matrix(self, time_series):
        """
        Calculates the distance matrix between all rows in the time series.
        
        Args:
            time_series (np.array): The time series to calculate the distance matrix from.

        Returns:
            np.array: The calculated distance matrix.
        """
        time_series_matrix = time_series.reshape(-1, 1)
        distances = pdist(time_series_matrix, metric='euclidean')
        distance_matrix = squareform(distances)
        return distance_matrix
    

    def normalize_and_scale_matrix(self, distance_matrix):
        """
        Normalizes and scales a distance matrix to a 0-255 range.
        
        Args:
            distance_matrix (np.array): The distance matrix to normalize and scale.

        Returns:
            np.array: The normalized and scaled distance matrix.
        """
        normalized_matrix = (distance_matrix - np.min(distance_matrix)) / (np.max(distance_matrix) - np.min(distance_matrix))
        scaled_matrix = (normalized_matrix * 255).astype(np.uint8)
        return scaled_matrix

    def merge_distance_matrices(self, *distance_matrices):
        """
        Merges multiple distance matrices into a single RGB image.
        
        Args:
            *distance_matrices (np.array): The distance matrices to merge.

        Returns:
            np.array: The merged RGB image.
        """
        # Check that there are at most 3 distance matrices
        if len(distance_matrices) > 3:
            raise ValueError("Can merge at most 3 distance matrices into an RGB image.")

        # Normalize and scale all distance matrices
        scaled_matrices = [self.normalize_and_scale_matrix(dm) for dm in distance_matrices]

        # Create an empty RGB image with the same shape as the first distance matrix
        height, width = distance_matrices[0].shape
        merged_image = np.zeros((height, width, 3), dtype=np.uint8)

        # Assign each scaled distance matrix to a channel in the RGB image
        for i, scaled_matrix in enumerate(scaled_matrices):
            merged_image[:, :, i] = scaled_matrix

        return merged_image

    def save_image(self, image, filename):
        """
        Saves an RGB image to a file.
        
        Args:
            image (np.array): The RGB image to save.
            filename (str): The filename to save the image to.
        """
        Image.fromarray(image).save(filename)

    def process_row(self, i, *dataframes):
        """
        Processes a row from each dataframe by calculating, merging, and saving the distance matrices.
        
        Args:
            i (int): The index of the row to process.
            *dataframes (pd.DataFrame): The dataframes to process the row from.
        """
        # Select the current row from each dataframe
        current_rows = [df.iloc[i] for df in dataframes]

        # Calculate the distance matrix from each current row
        distance_matrices = [self.calculate_distance_matrix(row.values.reshape(-1, 1)) for row in current_rows]

        # Merge the distance matrices into a single RGB image
        merged_image = self.merge_distance_matrices(*distance_matrices)

        # Save the merged image
        self.save_image(merged_image, f'colorimage/{i}.png')

    def process_dataframes(self, num_cores, *dataframes):
        """
        Processes all rows from each dataframe in parallel.
        
        Args:
            num_cores (int): The number of cores to use for parallel processing.
            *dataframes (pd.DataFrame): The dataframes to process.
        """
        Parallel(n_jobs=num_cores)(
            delayed(self.process_row)(i, *dataframes) for i in range(len(dataframes[0]))
        )

def main():
    """The main function to execute the time series analysis."""
    # Define the data files
    data_files = ["simu_Ba_r_T.csv", "simu_Ba_i_T.csv", "simu_Ma_i_T.csv"]

    # Load the data from the data files
    dataframes = [pd.read_csv(df_file, header=0) for df_file in data_files]

    # Create a TimeSeriesAnalyzer
    analyzer = TimeSeriesAnalyzer()

    # Process the dataframes
    analyzer.process_dataframes(1, *dataframes)

if __name__ == "__main__":
    main()
