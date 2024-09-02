import pandas as pd
import numpy as np
import ramanspy as rp

# Load data from a text file using pandas
data = pd.read_csv('PE_macro_785nm_1.4mW_400g_1x100_range1.txt', delimiter=';', header=None)

# Separate wavenumbers and intensities from the data
wavenumbers = data.iloc[:, 0].values
intensities = data.iloc[:, 1].values

# Create a Ramanspy spectrum object
spectrum = rp.Spectrum(intensities, wavenumbers)

# Apply a preprocessing pipeline with debug print statements
pipeline = rp.preprocessing.Pipeline([
    rp.preprocessing.misc.Cropper(region=(700, 1800)),
    rp.preprocessing.despike.WhitakerHayes(),
    rp.preprocessing.denoise.SavGol(window_length=9, polyorder=3),
    rp.preprocessing.baseline.ASPLS(),
    rp.preprocessing.normalise.MinMax()
])

# Apply preprocessing pipeline with debug print statements
preprocessed_spectrum = pipeline.apply(spectrum)


# Perform spectral unmixing
'''nfindr = rp.analysis.unmix.NFINDR(n_endmembers=5)
amaps, endmembers = nfindr.apply(preprocessed_spectrum)

# Plot results
rp.plot.spectra(endmembers)
rp.plot.image(amaps)
rp.plot.show()'''

preprocessed_spectrum.plot()
