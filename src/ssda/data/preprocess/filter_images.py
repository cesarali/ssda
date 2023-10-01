import os
import pickle
import numpy as np
from typing import List,Tuple
from dataclasses import dataclass
import numpy as np
from scipy.interpolate import interp1d

@dataclass
class NDS:
    name:str
    frequencies: np.array
    max_frequency:float
    min_frequency:float
    number_of_frequencies:int
    delta_frequency:float

    z: np.array
    max_z:float
    min_z:float
    number_of_z:int
    delta_z:float

    frequencies_to_id:dict = None
    z_domain:Tuple = None

    def __post_init__(self):
        self.frequencies_to_id = dict(zip(self.frequencies, range(self.number_of_frequencies)))
        self.z_to_id = dict(zip(self.z, range(self.number_of_z)))
        self.z_domain = (self.min_z, self.max_z, self.delta_z)

def reduce_image_frequency(image:np.array,
                           image_nds_dataclass:NDS,
                           selected_frequencies:List[float],
                           filter_nan=True):
    """
    real images are sampled at higher frequency, so we reduce the image to the
    ones of the selected images

    parameters
    ----------

    real_image: image to reduce
    simulations_nds:
    image_nds_dataclass:

    """
    accepted_frequencies = []
    reduced_image = []

    for frequency_to_find in selected_frequencies:

        id_of_frequency = image_nds_dataclass.frequencies_to_id[frequency_to_find]
        frequency_vs_z = image[id_of_frequency]
        if filter_nan:
            if not np.isnan(frequency_vs_z).any():
                reduced_image.append(frequency_vs_z)
                accepted_frequencies.append(frequency_to_find)
        else:
            reduced_image.append(frequency_vs_z)
            accepted_frequencies.append(frequency_to_find)

    reduced_image = np.vstack(reduced_image)

    return reduced_image, accepted_frequencies

def interpolate_image(simulations_image,
                      selected_z,
                      filter_z_in_simulation,
                      min_z_id_simulations,
                      max_z_id_simulations):
    filter_by_z_simulation_image = simulations_image[:, min_z_id_simulations:max_z_id_simulations]
    interpolated_image = []
    number_of_image_frequencies = filter_by_z_simulation_image.shape[0]

    for frequency_id in range(number_of_image_frequencies):
        # frequency
        frequency_values = filter_by_z_simulation_image[frequency_id]
        interp_function = interp1d(filter_z_in_simulation, frequency_values, kind='linear',
                                   fill_value="extrapolate")

        # Use the interpolation function to find corresponding y values
        frequency_interpolated = interp_function(selected_z)
        interpolated_image.append(frequency_interpolated)

    interpolated_image = np.vstack(interpolated_image)
    return interpolated_image

if __name__=="__main__":
    from ssda.data.preprocess.real import obtain_all_results

    #==================================
    # REAL DATA
    #==================================
    all_result_parent_folder = "D:/Projects/Clinical_Studies/CortBS_DEGUM_2022/06_Results/"
    full_data = obtain_all_results(all_result_parent_folder)

    mat_result = full_data["101B_Cortbs_report_no1.mat"]
    real_dz = mat_result.dz
    real_frequencies = mat_result.f_sampling
    real_image = mat_result.Ydiff

    dfrequencies_real = (real_frequencies[1:] - real_frequencies[:-1])[0]
    deltaz_real = (real_dz[1:] - real_dz[:-1])[0]

    real_nds = NDS(max_frequency=max(real_frequencies),
                   name="real",
                   min_frequency=min(real_frequencies),
                   number_of_frequencies=len(real_frequencies),
                   delta_frequency=dfrequencies_real,
                   frequencies=real_frequencies,
                   z=real_dz,
                   max_z=max(real_dz),
                   min_z=min(real_dz),
                   number_of_z=len(real_dz),
                   delta_z=deltaz_real)

    #======================================
    # SIMULATIONS REAL DATA
    #======================================
    from ssda import data_path
    import scipy
    from pathlib import Path

    raw_path = os.path.join(data_path, "raw")
    data_path = Path(raw_path)
    mat_path = list(data_path.glob("*.mat"))[0]
    mat = scipy.io.loadmat(mat_path, squeeze_me=True, struct_as_record=False)
    simulations_z = mat["z"]
    simulations_frequencies = mat["f"]
    all_images = mat["NDS"].transpose(2, 0, 1)
    simulation_image = all_images[0]

    dfrequencies_simulations = (simulations_frequencies[1:] - simulations_frequencies[:-1])[0]
    deltaz_simulations = (simulations_z[1:] - simulations_z[:-1])[0]

    simulations_nds = NDS(max_frequency=max(simulations_frequencies),
                          name="simulation",
                          min_frequency=min(simulations_frequencies),
                          number_of_frequencies=len(simulations_frequencies),
                          delta_frequency=dfrequencies_simulations,
                          frequencies=simulations_frequencies,
                          z=simulations_z,
                          max_z=max(simulations_z),
                          min_z=min(simulations_z),
                          number_of_z=len(simulations_z),
                          delta_z=deltaz_simulations)

    print(real_nds)
    print(simulations_nds)

    #======================================
    # REDUCE FREQUENCIES
    #======================================

    #first select frequencies from simulation since they are smaller
    reduced_real_image, selected_frequencies = reduce_image_frequency(real_image, real_nds, simulations_nds.frequencies)
    reduced_real_images_by_frequency = {}
    for name_, mat_result in full_data.items():
        real_image = mat_result.Ydiff
        reduced_real_image, _ = reduce_image_frequency(real_image, real_nds, selected_frequencies)
        reduced_real_images_by_frequency[name_] = reduced_real_image

    reduced_simulations_images_by_frequency = {}
    simulation_image = all_images[0]
    for image_id, simulation_image in enumerate(all_images):
        reduced_simulation_image, selected_frequencies_s = reduce_image_frequency(simulation_image, simulations_nds,selected_frequencies, False)
        reduced_simulations_images_by_frequency[image_id] = reduced_simulation_image

    #======================================
    # INTERPOLATION
    #======================================

    # the selected z is in the real domain
    selected_z = real_nds.z[4:]
    selected_min_z = min(selected_z)
    selected_max_z = max(selected_z)

    final_reduced_real_image = {}
    for name_, reduced_real_by_frequency in reduced_real_images_by_frequency.items():
        final_reduced_real_image[name_] = reduced_real_by_frequency[:, 4:]

    is_bigger = simulations_nds.z >= selected_min_z
    is_smaller = simulations_nds.z <= selected_max_z

    min_z_id_simulations = min(np.where(is_bigger)[0])
    max_z_id_simulations = max(np.where(is_smaller)[0]) + 2

    filter_z_in_simulation = simulations_nds.z[min_z_id_simulations:max_z_id_simulations]
    simulations_image = reduced_simulations_images_by_frequency[0]

    final_reduced_simulation_image = {}
    for image_id,simulations_image in reduced_simulations_images_by_frequency.items():
        final_reduced_simulation_image[image_id] = interpolate_image(simulations_image,
                                                                     selected_z,
                                                                     filter_z_in_simulation,
                                                                     min_z_id_simulations,
                                                                     max_z_id_simulations)

    from ssda import data_path

    simulation_filtered_path = os.path.join(data_path,"preprocessed","filtered_simulation_images.pkl")
    pickle.dump(final_reduced_simulation_image,open(simulation_filtered_path,"wb"))

    real_filtered_path = os.path.join(data_path,"preprocessed","filtered_real_images.pkl")
    pickle.dump(final_reduced_real_image,open(real_filtered_path,"wb"))