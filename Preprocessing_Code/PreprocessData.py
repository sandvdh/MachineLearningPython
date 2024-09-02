# Standard libraries
import os
import pandas as pd
import numpy as np
import scipy as sp  
import typing

import matplotlib.pyplot as plt

# Own imports
import Functionality

# User specified parameters
_Despiking_Types = typing.Literal["WhitHay", "None"]
_Denoising_Types = typing.Literal["Huber_D", "Huber_D2", "SavGol", "None"]
_Normalization_Types = typing.Literal["L2", "L1", "MinMax"]

# Main function
def compile_raw_csv_to_preprocessed_csv(input_directory: str, output_directory: str, input_file: str = "", 
                                        output_file: str = "compiled_preprocessed_data", output_format: str = "csv",
                                        recursive: bool = False, outputframe: pd.DataFrame = pd.DataFrame(),
                                        despiking_algorithm: _Despiking_Types = "WhitHay", 
                                        denoising_algorithm: _Denoising_Types = "SavGol",
                                        normalization: _Normalization_Types = "L2"):
    """
    If the filename input_file is left empty, then every file with the format input_format
    """

    # Create the output directory if it doesn't exist
    os.makedirs(output_directory, exist_ok=True)

    if not input_file:
        if recursive:
            for root, dirs, files in os.walk(input_directory):
                for filename in files:
                    if filename.endswith(".csv") or filename.endswith(".xlsx"):
                        outputframe = compile_raw_csv_to_preprocessed_csv(input_directory, output_directory, 
                                                            filename, output_file, output_format,
                                                            False, outputframe, despiking_algorithm,
                                                            denoising_algorithm, normalization)
        else:
            for filename in os.listdir(input_directory):
                if filename.endswith(".csv") or filename.endswith(".xlsx"):
                    outputframe = compile_raw_csv_to_preprocessed_csv(input_directory, output_directory, 
                                                            filename, output_file, output_format,
                                                            False, outputframe, despiking_algorithm,
                                                            denoising_algorithm, normalization)
        return
    
    if (not input_file.endswith(".csv")) and (not input_file.endswith(".xlsx")):
        input_file = input_file + "." + output_format
    
    inputframe = pd.read_csv(os.path.join(input_directory, input_file), encoding='latin1')
    outputframe = pd.concat([inputframe, outputframe])
    wavenumbers = list(inputframe.select_dtypes('number'))
    inputframe = inputframe[wavenumbers]

    for idx in range(inputframe.shape[0]):
        spectrum = inputframe.iloc[idx]
        spectrum = spectrum.to_numpy().flatten()
        spectrum = (spectrum - np.median(spectrum))/(np.linalg.norm(spectrum, 1) - np.median(spectrum))
        
        if any(np.isnan(spectrum)):
            print(spectrum)
            break

        baseline = Functionality.asPLS(spectrum, 1e6, 1e-5, 1e5)
        basedspec = spectrum - baseline
        match denoising_algorithm:
            case "Huber_D":
                denspec = Functionality.ChambollePock(basedspec, Functionality.apply_D, Functionality.apply_D_conjugate, 4, 2.5e1, 2.5e0, 1e-10, 1e5)[0]
            case "Huber_D2":
                denspec = Functionality.ChambollePock(basedspec, Functionality.apply_D2, Functionality.apply_D2_conjugate, 4, 2.5e1, 5e-1, 1e-10, 1e5)[0]
            case "SavGol":
                denspec = sp.signal.savgol_filter(basedspec, 9, 3)
            case "None":
                denspec = basedspec
            case _:
                print("\""+denoising_algorithm+"\" is not a valid denoising algorithm.\nperhaps try another option.")
                denspec = basedspec

        match normalization:
            case "L2":
                df_spectrum = (denspec - np.mean(denspec))/(np.linalg.norm(denspec - np.mean(denspec), 2))
            case "L1":
                df_spectrum = (denspec - np.median(denspec))/(np.linalg.norm(denspec - np.median(denspec), 1))
            case "MinMax":
                df_spectrum = (denspec - np.min(denspec))/(np.max(denspec) - np.min(denspec))
            case _:
                print("\""+normalization+"\" is not a valid normalization.\nperhaps try another option.")
                df_spectrum = denspec

        outputframe.loc[outputframe.shape[0]-inputframe.shape[0]+idx, wavenumbers] = df_spectrum

    output_file = output_file + '.' + output_format

    match output_format:
        case 'csv':
            outputframe.to_csv(os.path.join(output_directory, output_file), index=False)
            print("Combined data saved to csv.")
        case 'xlsx':
            outputframe.to_excel(os.path.join(output_directory, output_file), index=False)
            print("Combined data saved to Excel.")
        case _:
            print("Output format should be csv or xlsx.")

    return outputframe

if __name__ == "__main__":
    compile_raw_csv_to_preprocessed_csv('data/LeiEtAl', 'data/LeiEtAlPreprocessedCSV', despiking_algorithm='None', denoising_algorithm='Huber_D2', normalization='MinMax')