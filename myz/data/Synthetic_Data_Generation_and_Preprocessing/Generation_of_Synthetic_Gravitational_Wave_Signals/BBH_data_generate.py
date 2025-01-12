import random
import lalsimulation as lalsim
import lal
import bilby
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.interpolate import interp1d
from bilby_cython import geometry


# Geocentric time ranges for O1, O2, O3a, and O3b
geocentric_time_ranges = {
    "O1": (1126051217, 1137254417),
    "O2": (1164556817, 1187733618),
    "O3a": (1238166018, 1253977218),
    "O3b": (1256655618, 1269363618),
}


def get_random_geocentric_time():
    #Choose a random observation period
    period = random.choice(list(geocentric_time_ranges.keys()))
    start, end = geocentric_time_ranges[period]
    #Choose a random GPS time from this period
    return random.uniform(start, end)

def calculate_strain(detector, h_plus, h_cross, plus_polarization_tensor, cross_polarization_tensor):
    f_plus = np.einsum('ij,ij->', detector.detector_tensor, plus_polarization_tensor)
    f_cross = np.einsum('ij,ij->', detector.detector_tensor, cross_polarization_tensor)
    return f_plus * h_plus + f_cross * h_cross

def interpolate_strain(strain, strain_time, geocent_time, sampling_frequency, duration):
    strain_detector_time = strain_time + geocent_time
    n = sampling_frequency * duration
    data_start_time = int(geocent_time) - pre_trigger_duration
    data_detector_time = np.arange(n) / sampling_frequency + data_start_time
    h_interp = interp1d(strain_detector_time, strain, fill_value=0, bounds_error=False)(data_detector_time)
    return data_detector_time, h_interp



# Parameters
solar_mass = bilby.core.utils.constants.solar_mass
sampling_frequency = 4096
deltaT = 1 / sampling_frequency
duration = 32
post_trigger_duration = 0.5
pre_trigger_duration = duration - post_trigger_duration

# detectors  (we only use L1)
#H1 = bilby.gw.detector.get_empty_interferometer("H1")
L1 = bilby.gw.detector.get_empty_interferometer("L1")
#V1 = bilby.gw.detector.get_empty_interferometer("V1")

num_samples = 1000 
data = []  

for i in range(num_samples):
    geocent_time = get_random_geocentric_time()
    mass_1 = np.random.uniform(2, 95) * solar_mass
    mass_2 = np.random.uniform(2, min(mass_1 / 10, 95)) * solar_mass
    spin_1z = np.random.uniform(-0.70, 0.70)
    spin_2z = np.random.uniform(-0.70, 0.70)
    spin_1x, spin_1y = 0, 0
    spin_2x, spin_2y = 0, 0
    luminosity_distance = np.random.uniform(50, 500)
    theta_jn = np.random.uniform(0, np.pi)
    phase = np.random.uniform(0, 2 * np.pi)
    longAscNodes = 0
    eccentricity = 0
    meanPerAno = 0
    LALParams = lal.CreateDict()
    waveform_approximant = "SEOBNRv4"
    approximant = lalsim.GetApproximantFromString(waveform_approximant)


    f_min = max(20, 0.95 * lalsim.SimInspiralChirpStartFrequencyBound(pre_trigger_duration, mass_1, mass_2))
    if lalsim.SimInspiralGetSpinFreqFromApproximant(approximant) == lalsim.SIM_INSPIRAL_SPINS_FLOW:
        f_ref = f_min
    else:
        f_ref = 20


    # Generating Waves
    h_plus_timeseries, h_cross_timeseries = lalsim.SimInspiralChooseTDWaveform(
        mass_1, mass_2, spin_1x, spin_1y, spin_1z, spin_2x, spin_2y, spin_2z,
        luminosity_distance, theta_jn, phase, longAscNodes, eccentricity, meanPerAno,
        deltaT, f_min, f_ref, LALParams, approximant
    )


    h_plus = h_plus_timeseries.data.data
    h_cross = h_cross_timeseries.data.data
    h_plus_time = np.arange(len(h_plus)) * h_plus_timeseries.deltaT + float(h_plus_timeseries.epoch)


    # Detector Projection
    ra = np.random.uniform(0, 2 * np.pi)  # right ascension
    dec = np.random.uniform(-np.pi / 2, np.pi / 2)  # Declination
    psi = np.random.uniform(0, np.pi)  # Polarization angle


    plus_polarization_tensor = geometry.get_polarization_tensor(ra, dec, geocent_time, psi, "plus")
    cross_polarization_tensor = geometry.get_polarization_tensor(ra, dec, geocent_time, psi, "cross")


    #strain_H1 = calculate_strain(H1, h_plus, h_cross, plus_polarization_tensor, cross_polarization_tensor)
    strain_L1 = calculate_strain(L1, h_plus, h_cross, plus_polarization_tensor, cross_polarization_tensor)
    #strain_V1 = calculate_strain(V1, h_plus, h_cross, plus_polarization_tensor, cross_polarization_tensor)


    # Interpolation
    #_, h_interp_H1 = interpolate_strain(strain_H1, h_plus_time, geocent_time, sampling_frequency, duration)
    _, h_interp_L1 = interpolate_strain(strain_L1, h_plus_time, geocent_time, sampling_frequency, duration)
    #_, h_interp_V1 = interpolate_strain(strain_V1, h_plus_time, geocent_time, sampling_frequency, duration)


    # data
    data.append({
        #"H1": h_interp_H1,
        "L1": h_interp_L1,
        #"V1": h_interp_V1,
        "parameters": {
            "mass_1": mass_1, "mass_2": mass_2, "spin_1z": spin_1z, "spin_2z": spin_2z,
            "luminosity_distance": luminosity_distance, "theta_jn": theta_jn,
            "ra": ra, "dec": dec, "psi": psi
        }
    })



#DATAFRAME:

rows = []

for sample in data:
    #h1_flat = sample["H1"].flatten()
    l1_flat = sample["L1"].flatten()
    #v1_flat = sample["V1"].flatten()
    
    row = {
        #**{f"H1_{i}": val for i, val in enumerate(h1_flat)},  # H1 
        **{f"L1_{i}": val for i, val in enumerate(l1_flat)},  # L1 
        #**{f"V1_{i}": val for i, val in enumerate(v1_flat)},  # V1 
        "label": 0  # BBH's label
    }
    rows.append(row)

df = pd.DataFrame(rows)

# save data
df.to_csv("/ayca/myz/data/1000_example/BBH_1000_4K.csv", index=False)

print("Data saved in CSV format: BBH_1000_4K.csv")



"""

# Visualize

plt.figure(figsize=(12, 8))
for i, signal in enumerate(data):
    plt.subplot(num_samples, 1, i + 1)
    plt.plot(signal["L1"], label=f"Sinyal {i + 1} - L1", color="blue")
    plt.legend()
plt.tight_layout()
plt.show()

"""