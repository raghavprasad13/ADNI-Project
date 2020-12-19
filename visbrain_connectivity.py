import numpy as np
import pandas as pd

from visbrain.gui import Brain
from visbrain.objects import SourceObj, ConnectObj

from os.path import join, exists
from glob import glob

# mmse_df = pd.read_csv(join('..', 'New_Data', 'AD', 'FDG', 'stats', 'output_mmse.csv'))
# npiq_df = pd.read_csv(join('..', 'New_Data', 'AD', 'FDG', 'stats', 'output_npiq.csv'))

# df = pd.concat([mmse_df, npiq_df], axis=1)
# df = df.loc[:,~df.columns.duplicated()]
# print(len(mmse_df))
# mmse_df = mmse_df[(mmse_df != 0).all(1)]
# print(len(mmse_df))

# print(mmse_df.head(1))
# print(len(df.columns))

dataset_path = join('..', 'New_Data')
diagnoses = glob(join(dataset_path, '*'))
diagnoses = list(filter(lambda path: 'stats' not in path, diagnoses))

percolation_states = {'AD': [],
                      'MCI': [],
                      'CN': []}

for diagnosis in diagnoses:
    diagnosis_name = diagnosis.split('/')[-1]
    scans = glob(join(diagnosis, '*', '*'))
    scans = list(filter(lambda path: exists(join(path, 'percolation.npy')),
                        scans))
    for scan in scans:
        if ('Metadata' in scan) or ('stats' in scan):
            continue
        percolation_states[diagnosis_name].append(np.load(join(scan, 'percolation.npy')))

    percolation_states.update({diagnosis_name: np.nanmean(np.array(percolation_states[diagnosis_name]), axis=0)})

# exit()

scan = '131_S_0497~2006-06-28_12_44_51.0~I17585'
adj_mat = np.load(join('..', 'New_Data', 'AD', 'FDG', scan, 'adj_mat_thresh.npy'))
adj_mat[np.tril_indices_from(adj_mat)] = 0
adj_mat = np.absolute(adj_mat)
percolation = np.load(join('..', 'New_Data', 'AD', 'FDG', scan, 'percolation.npy'))

umin = 0
umax = 1

adj_mat = np.ma.masked_array(adj_mat, mask=True)
adj_mat.mask[np.where((adj_mat > umin) & (adj_mat < umax))] = False

# print(adj_mat.shape)
# print(percolation.shape)

atlas_df = pd.read_csv('Juelich_MMSE_NPIQ.csv')

loc_mat = np.array([[row['Voxel_X']-87, row['Voxel_Y']-120, row['Voxel_Z']-70] for index, row in atlas_df.iterrows()])
# print(loc_mat.shape)

cn_s_obj = SourceObj('rois', loc_mat, data=percolation_states['CN'], color='green', alpha=0.5, edge_width=1., radius_min=2., radius_max=15.)
# ad_s_obj = SourceObj('rois', loc_mat, data=percolation_states['AD'], color='green', alpha=1, edge_width=1., radius_min=2., radius_max=15.)
ad_s_obj = SourceObj('rois', loc_mat, color='green', alpha=1, radius_min=15, radius_max=15)
mci_s_obj = SourceObj('rois', loc_mat, data=percolation_states['MCI'], color='yellow', alpha=0.5, edge_width=1., radius_min=2., radius_max=15.)

c_obj = ConnectObj('edges', loc_mat, adj_mat, alpha=0.5, line_width=1.5,
                   cmap='gnuplot', antialias=True)

# func_net = nx.from_numpy_array(adj_mat)
# print(func_net.number_of_edges())
vb = Brain(source_obj=ad_s_obj, connect_obj=c_obj)

vb.show()
# # Create an empty kwargs dictionnary :
# kwargs = {}

# # ____________________________ DATA ____________________________

# # Load the xyz coordinates and corresponding subject name :
# # mat = np.load(download_file('xyz_sample.npz', astype='example_data'))
# mat = np.load(join('/', 'Users', 'raghavprasad', 'visbrain_data', 'example_data', 'xyz_sample.npz'))
# print(type(mat['xyz']))
# print(mat['xyz'])
# print(mat['xyz'].shape)
# print(type(mat['subjects']))
# print(mat['subjects'])
# print(mat['subjects'].shape)
# exit()
# xyz, subjects = mat['xyz'], mat['subjects']

# N = xyz.shape[0]  # Number of electrodes

# # Now, create some random data between [-50,50]
# data = np.random.uniform(-50, 50, len(subjects))

# """Create the source object :
# """
# s_obj = SourceObj('SourceObj1', xyz, data, color='crimson', alpha=.5,
#                   edge_width=2., radius_min=2., radius_max=10.)

# """
# To connect sources between them, we create a (N, N) array.
# This array should be either upper or lower triangular to avoid
# redondant connections.
# """
# connect = 1000 * np.random.rand(N, N)		# Random array of connections
# connect[np.tril_indices_from(connect)] = 0  # Set to zero inferior triangle

# """
# Because all connections are not necessary interesting, it's possible to select
# only certain either using a select array composed with ones and zeros, or by
# masking the connection matrix. We are giong to search vealues between umin and
# umax to limit the number of connections :
# """
# umin, umax = 30, 31

# # 1 - Using select (0: hide, 1: display):
# select = np.zeros_like(connect)
# select[(connect > umin) & (connect < umax)] = 1

# # 2 - Using masking (True: hide, 1: display):
# connect = np.ma.masked_array(connect, mask=True)
# connect.mask[np.where((connect > umin) & (connect < umax))] = False

# print('1 and 2 equivalent :', np.array_equal(select, ~connect.mask + 0))

# """Create the connectivity object :
# """
# c_obj = ConnectObj('ConnectObj1', xyz, connect, color_by='strength',
#                    dynamic=(.1, 1.), cmap='gnuplot', vmin=umin + .2,
#                    vmax=umax - .1, under='red', over='green',
#                    clim=(umin, umax), antialias=True)

# """Finally, pass source and connectivity objects to Brain :
# """
# vb = Brain(source_obj=s_obj, connect_obj=c_obj)

# vb.show()
