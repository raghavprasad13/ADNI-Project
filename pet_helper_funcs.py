import numpy as np
import nibabel as nib


def time_series_to_matrix(subject_time_series_path, parcel_path):
    '''
    Makes correlation matrix from parcel
    '''
    subject_time_series = nib.load(subject_time_series_path).get_fdata()
    parcel = nib.load(parcel_path).get_fdata().astype(int)
    g = np.zeros((np.max(parcel), subject_time_series.shape[-1]))
    for i in range(np.max(parcel)):
        g[i, :] = np.nanmean(subject_time_series[parcel == i+1], axis=0)

    np.nan_to_num(g, copy=False)

    h = np.mean(g, axis=1)
    return {'pre_adj': g, 'percolation': h}
