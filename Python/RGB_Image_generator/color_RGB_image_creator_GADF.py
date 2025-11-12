import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.preprocessing import StandardScaler
from PIL import Image
from pyts.image import GramianAngularField
import os

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
        Fits the scaler on the feature vector and then transforms it.      
        Args:
            feature_vector (np.array): The feature vector to fit and transform.       
        Returns:
            np.array: The fitted and transformed feature vector.
        """
        return self.scaler.fit_transform(feature_vector)
    
    def normalize_matrix(self, matrix):
        """
        Normalizes a matrix to the range [0, 1].        
        Args:
            matrix (np.array): The matrix to normalize.      
        Returns:
            np.array: The normalized matrix.
        """
        min_val = np.min(matrix)
        max_val = np.max(matrix)
        if min_val == max_val:
            return np.zeros_like(matrix)
        else:
            return (matrix - min_val) / (max_val - min_val)
    
    def calculate_gadf(self, time_series):
        """
        Calculates the Gramian Angular Summation Field (GASF) of a time series.       
        Args:
            time_series (np.array): The time series to transform.       
        Returns:
            np.array: The GASF of the time series.
        """
        gadf = GramianAngularField(method='difference')
        matrix_gadf = gadf.fit_transform(time_series.reshape(1, -1))
        return matrix_gadf[0]

    def normalize_and_scale_matrix(self, matrix):
        """
        Normalizes and scales a matrix to the range [0, 255].        
        Args:
            matrix (np.array): The matrix to normalize and scale.       
        Returns:
            np.array: The normalized and scaled matrix.
        """
        normalized_matrix = self.normalize_matrix(matrix)
        scaled_matrix = (normalized_matrix * 255).astype(np.uint8)
        return scaled_matrix

    def merge_gadf_matrices(self, *matrices):
        """
        Merges multiple GASF matrices into a single RGB image.
        
        Args:
            *matrices (np.array): The GASF matrices to merge.
        
        Returns:
            np.array: The merged RGB image.
        """
        # Ensure there are at most 3 matrices
        if len(matrices) > 3:
            raise ValueError("Can merge at most 3 GASF matrices into an RGB image.")

        # Normalize and scale all matrices
        scaled_matrices = [self.normalize_and_scale_matrix(matrix) for matrix in matrices]

        # Create an empty RGB image with the same shape as the first matrix
        height, width = scaled_matrices[0].shape
        merged_image = np.zeros((height, width, 3), dtype=np.uint8)

        # Assign each scaled matrix to a channel in the RGB image
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
        Processes a row from each dataframe by calculating, merging, and saving the GASF matrices.
        
        Args:
            i (int): The index of the row to process.
            *dataframes (pd.DataFrame): The dataframes to process the row from.
        """
        # Select the current row from each dataframe
        current_rows = [df.iloc[i].values for df in dataframes]

        # Reshape each row into a 2D array and calculate the GASF
        gadf_matrices = [self.calculate_gadf(row) for row in current_rows]

        # Merge the GASF matrices into a single RGB image
        merged_image = self.merge_gadf_matrices(*gadf_matrices)

        # Ensure directory exists before saving
        os.makedirs('colorimage', exist_ok=True)
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
    # Define the data directory
    data_dir = 'C:/Users/truon011/OneDrive - Wageningen University & Research/Wageningen UR _ Research_Work/Python codes/Quentin codes/Data/Data_check/Quentin_data_split'
    os.chdir(data_dir)
    
    # Define the data files
    data_files = ["df_R_oiseau.csv", "df_I_oiseau.csv", "df_I_moustique.csv"]

    # Load the data from the data files
    dataframes = [pd.read_csv(df_file, header=0) for df_file in data_files]

    # Create a TimeSeriesAnalyzer
    analyzer = TimeSeriesAnalyzer()

    # Process the dataframes
    analyzer.process_dataframes(1, *dataframes)

if __name__ == "__main__":
    main()
