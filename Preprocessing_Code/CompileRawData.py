import os
import pandas as pd
import numpy as np
import Functionality
from scipy.interpolate import CubicSpline


def interpolate_truncate_spectra(intensities, wavenumbers, interpolated_wavenumbers: np.ndarray = np.arange(720, 1801, 1)) -> np.ndarray:
    """
    Reads and interpolates spectra from files in the input directory and save the interpolated spectra
    in the output directory.

    Args:
    input_directory (str): Path to the input directory containing spectra files.
    output_directory (str): Path to the output directory to save the interpolated spectra.
    """

    # Read and interpolate spectra from files in the input directory
    # Read spectrum data
    

    # Perform cubic spline interpolation
    spline_function = CubicSpline(wavenumbers, intensities)

    # Generate interpolated wavenumbers (deafult is from 720 to 1800 with a spacing of 1)
    interpolated_intensities = spline_function(interpolated_wavenumbers)

    # Return result
    return interpolated_intensities
    


def extract_info_from_filename(filename):
    """
    Extract plastic name, size, laser, power, grating, and acquisition time scan number from the filename.
    """
    parts = filename.split(' ')
    plastic_name = parts[0]
    size = next((part for part in parts if 'micro' in part or 'macro' in part), None)
    laser = next((part for part in parts if 'nm' in part), None)
    power = next((part for part in parts if 'mW' in part), None)
    grating = next((part for part in parts if 'g' in part), None)
    acq_time_scan_nmb = next((part for part in parts if 'x' in part and part.replace('x', '').isdigit()), None)

    return filename, plastic_name, size, laser, power, grating, acq_time_scan_nmb


# TODO: Make more general - seperate compiling function from naming function.
def compile_txt_directory_to_csv(input_directory: str, output_directory: str, output_filename: str = "compiled_raw_data", format: str = "csv", wavenumbers: np.ndarray = np.arange(720, 1801, 1), text_delimiter: str = ';') -> None:
    """, 
    This function creates a csv file from a directory containing .txt measurement files.
    The files must first be interpolated and truncated to identical wavenumber datapoints.

    If the .txt files follow the following naming convention:
        (plastic_name)_(micro/macro)_(laser_wavelength)nm_(laser_power)mW_(grating)g_(time)x.txt
    where the items between brackets are filled in accordingly, the information in brackets is also extracted into named columns.

    Args:
    input_directory (str): Path to the directory containing the .txt files.
    output_directory (str): Path to the directory where the compiled file will be saved.
    output_filename (str - default: "copmiled_raw_data"): name of the compiled file,
    format (str - default: "csv"): format of the compiled file. Options are csv (for .csv) or xlsx (for Excel .xlsx)
    """
    # Create the output directory if it doesn't exist
    os.makedirs(output_directory, exist_ok=True)

    # Iterate over files in the input directory
    intensities = np.zeros([0, len(wavenumbers)])
    metadata_data = []
    Idx = 0
    for root, dirs, files in os.walk(input_directory):
        # Compute the interpolated intensity values
        for filename in files:
            if filename.endswith('.txt') and 'range2' not in filename:
                # Read out wavenumbers and intensities
                file_path = os.path.join(root, filename)
                spectrum_data = pd.read_table(file_path, delimiter=text_delimiter, header=None)
                print(spectrum_data)
                original_wavenumbers = spectrum_data.iloc[:, 0].to_numpy()
                original_intensities = spectrum_data.iloc[:, 1].to_numpy()
                
                # despike -!-the original-!- spectra (do not interpolate through peaks)
                original_intensities = Functionality.WhitakerHayes(y = original_intensities, neighbours = 4, threshold = 7)
                original_intensities = interpolate_truncate_spectra(original_intensities, original_wavenumbers, wavenumbers)

                # Add intensities to 
                intensities = np.vstack([intensities, original_intensities])

                # Extract plastic name, size, laser, power, grating, and acquisition time scan number from filenames
                metadata_data.append(extract_info_from_filename(filename))

    # Initialize dataframes to hold the combined data and metadata
    output_data = pd.DataFrame(data = intensities, columns = wavenumbers)
    metadata = pd.DataFrame(data = metadata_data, columns =["Filename", "Plastic", "Size", "Laser", "Power", "Grating", "Acq_Time_Scan_Nmb"])

    # Combine dataframes into a single dataframe
    dataframe = pd.concat([metadata, output_data], axis=1)

    # Prompt user for column removal
    remove_columns = input("Do you want to remove filename, size, laser, power, grating, and acquisition time scan number columns? (y/n): ")
    while True:
        match remove_columns.lower():
            case 'y':
                dataframe = dataframe.drop(columns=['Filename', 'Size', 'Laser', 'Power', 'Grating', 'Acq_Time_Scan_Nmb'])
                break
            case 'n':
                break
            case _:
                print("Please answer y (yes) or n (no):")

    # Reorder the columns
    dataframe = dataframe[['Plastic'] + [col for col in dataframe.columns if col != 'Plastic']]

    # Write the combined data to file
    output_file = output_filename + '.' + format
    match format:
        case 'csv':
            dataframe.to_csv(os.path.join(output_directory, output_file), index=False)
            print("Combined data saved to csv.")
        case 'xlsx':
            dataframe.to_excel(os.path.join(output_directory, output_file), index=False)
            print("Combined data saved to Excel.")
        case _:
            print("Format should be csv or xlsx.")