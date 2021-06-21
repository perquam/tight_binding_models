import utils
from CF import config_parameters as c_p

# All important objects
file_manager = utils.ManagingFiles(directory_save='results', directory_load='results')
data_generator = utils.GeneratingData( c_p.a0_bounds, c_p.Vd2sigma_bounds, c_p.Vd2pi_bounds,
                                       c_p.Vd2delta_bounds, c_p.e0_bounds, c_p.e1_bounds, c_p.e2_bounds,
                                       file_manager )
dataset_filename = 'data_10_a_const.npy'

# clear data in a file
file_manager.erase_data(dataset_filename)

# generate data
materials_dataset, bands_dataset = data_generator.generate_data(size_of_data = 10)

# shifting bands
for bands, materials in zip(bands_dataset, materials_dataset):
    offset = utils.np.amax(bands[:,0])
    bands -= offset
    materials.e0 -= offset
    materials.e1 -= offset 
    materials.e2 -= offset

# normalize energies to (0, 1), calculate means and standard deviations
E_min, E_max = utils.cf.normalize_band(bands_dataset)
means = utils.cf.get_means(bands_dataset)
standard_deviations = utils.cf.get_standard_deviations(bands_dataset, means)
print('E_min: ' + str(E_min) + ' E_max: ' + str(E_max) + ' means: ' + str(means)
      + ' std devs: ' + str(standard_deviations))

# save data
file_manager.save_bands_dataset(materials_dataset, bands_dataset, dataset_filename)

# load data
# file_manager.load_bands_dataset(dataset_filename)

# Script - comparing to real data
#MoS2_3_params, MoS2_3_bands = file_manager.load_and_interpolate_bands('MoS2_3_bands.csv', interpolate_from=65, interpolate_to=166)
#utils.cf.normalize_band(MoS2_3_bands, E_min, E_max) # !!! The real data has to be normalized, relative to the training data.
#file_manager.save_bands_dataset(MoS2_3_params, MoS2_3_bands, 'MoS2_3_bands.npy')
#MoS2_3_params_output, MoS2_3_bands_output = file_manager.load_bands_dataset('MoS2_3_bands_output.npy')
#utils.cf.compare_output_vs_target(MoS2_3_params_output[0], MoS2_3_params[0], MoS2_3_bands_output[0], MoS2_3_bands[0])

# Script - showing the generated results
for i in range(0, 10):
    utils.cf.plot_band(materials_dataset[i], bands_dataset[i])