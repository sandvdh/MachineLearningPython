import os
import numpy as np
import pandas as pd
#from Preprocessing import *
import Functionality
import CompileRawData
import PreprocessData
from sklearn.preprocessing import RobustScaler

def preprocess_spectrum(input_directory,
                        temp_directory,
                        output_directory,
                        wavenumbers: np.ndarray = np.arange(720, 1801, 1),
                        despiking_algorithm: str = "WhitHay", 
                        denoising_algorithm: str = "SavGol",
                        normalization: str = "L2",
                        text_delimiter: str = ';'):
    """
    Preprocess spectrum data by interpolating and truncating spectra and saving in temp_directory, then iterating over each individual spectrum in this directory and
    performing baseline reduction, Savitzky-Golay denoising, and feature selection; and saving the processed spectra in a new output_directory.

    Args:
    input_directory (str): Path to the input directory containing raw spectra files.
    temp_directory (str): Path to the temporary directory to store intermediate files.
    output_directory (str): Path to the output directory to save the preprocessed data.

    Returns:
    None
    """
    # Interpolate and truncate spectra, after despiking
    CompileRawData.compile_txt_directory_to_csv(input_directory,
                                                temp_directory,
                                                "compiled_raw_data",
                                                "csv",
                                                wavenumbers,
                                                text_delimiter)

    # Data augmentation
    #AugmentData

    # Process each spectrum in the temp directory
    PreprocessData.compile_raw_csv_to_preprocessed_csv(temp_directory,
                                                       output_directory,
                                                       "compiled_raw_data", 
                                                       "compiled_preprocessed_data", 
                                                       "csv", 
                                                       False, 
                                                       pd.DataFrame(),
                                                       despiking_algorithm,
                                                       denoising_algorithm, 
                                                       normalization)



if __name__ == "__main__":
    preprocess_spectrum('data/RawTxt', 'data/RawCSV', 'data/PreprocessedCSV')
    #preprocess_spectrum('data/SLoPP-E', 'data/RawCSVSLoPP-E', 'data/PreprocessedCSVSLoPP-E', text_delimiter="\t")