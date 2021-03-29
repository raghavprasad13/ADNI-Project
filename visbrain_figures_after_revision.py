import numpy as np
import pandas as pd

from visbrain.gui import Brain
from visbrain.objects import SourceObj, ConnectObj

from os.path import join
from glob import glob

dataset_path = join('..', 'Data_revision')

atlas_df = pd.read_csv('Juelich_MMSE_NPIQ.csv')

stats = join(dataset_path, 'stats')

anova_av45_df = pd.read_csv(join(stats, 'anova_av45.csv'))
av45_eligible_rois = anova_av45_df.loc[anova_av45_df['p_value'] < 0.05]
av45_eligible_rois = av45_eligible_rois['ROI'].tolist()

loc_mat_av45 = np.array([[row['Voxel_X']-87,
                          row['Voxel_Y']-120,
                          row['Voxel_Z']-70]
                        for index, row in atlas_df.iterrows()
                        if row['Anatomical Region'] in av45_eligible_rois])

anova_pib_df = pd.read_csv(join(stats, 'anova_pib.csv'))
pib_eligible_rois = anova_pib_df.loc[anova_pib_df['p_value'] < 0.05]
pib_eligible_rois = pib_eligible_rois['ROI'].tolist()

loc_mat_pib = np.array([[row['Voxel_X']-87,
                         row['Voxel_Y']-120,
                         row['Voxel_Z']-70]
                        for index, row in atlas_df.iterrows()
                        if row['Anatomical Region'] in pib_eligible_rois])

av45_src_obj = SourceObj('rois', loc_mat_av45, color='green', alpha=1,
                         radius_min=12, radius_max=12)

pib_src_obj = SourceObj('rois_pib', loc_mat_pib, color='blue', alpha=1,
                        radius_min=12, radius_max=12)

vb = Brain(source_obj=[av45_src_obj, pib_src_obj], bgcolor='white')
vb.show()
