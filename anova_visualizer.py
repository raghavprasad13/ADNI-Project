"""
Add deep sources
================

Add sources to the scene. This script also illustrate most of the controls for
sources. Each source is defined by a (x, y, z) MNI coordinate. Then, we can
attach some data to sources and project this activity onto the surface
(cortical projection). Alternatively, you can run the cortical repartition
which is defined as the number of contributing sources per vertex.

.. image:: ../../_static/examples/ex_sources.png
"""
import numpy as np

from visbrain.gui import Brain
from visbrain.objects import SourceObj, BrainObj

from os.path import join
import pandas as pd

kwargs = {}
RADIOISOTOPES = ['AV45', 'FDG', 'PiB']
RADIOISOTOPES_INDICES = {'AV45': 0, 'FDG': 1, 'PiB': 2}
CRITICAL_F = {'AV45': 3.005, 'FDG': 3.00053, 'PiB': 3.042}

stats_path = join('..', 'New_Data', 'stats')

anova_dfs = {radioisotope: pd.read_csv(join(stats_path, 'anova_'+radioisotope.lower()+'.csv')) for radioisotope in RADIOISOTOPES}
anova_dfs.update({radioisotope: anova_dfs[radioisotope][anova_dfs[radioisotope]['f_value'] >= CRITICAL_F[radioisotope]] for radioisotope in RADIOISOTOPES})

anova_rois = {radioisotope: anova_dfs[radioisotope]['ROI'].tolist() for radioisotope in RADIOISOTOPES}
rois_radioisotope = np.array([radioisotope for radioisotope in anova_rois for i in range(len(anova_rois[radioisotope]))])
rois = np.array([roi for radioisotope in anova_rois for roi in anova_rois[radioisotope]])
print(len(rois_radioisotope))
print(rois_radioisotope)
print(len(rois))
print(rois)

atlas_df = pd.read_csv('Juelich_MMSE_NPIQ.csv')
atlas_df = atlas_df[atlas_df['Anatomical Region'].isin(rois)]
# print(atlas_df)
# print(len(atlas_df))

loc_mat = np.array([[row['Voxel_X']-85, row['Voxel_Y']-120, row['Voxel_Z']-70] for index, row in atlas_df.iterrows()])
# exit()

"""The "subjects" list is composed of 6 diffrents subjects and here we set one
unique color (u_color) per subject.
"""
u_color = ['green', 'red', 'blue']
kwargs['color'] = [u_color[RADIOISOTOPES_INDICES[radioisotope]] for radioisotope in rois_radioisotope]
kwargs['alpha'] = 0.7

"""
Now we attach data to each source.
"""
kwargs['data'] = np.arange(len(rois_radioisotope))

"""The source's radius is proportional to the data attached. But this
proportion can be controlled using a minimum and maximum radius
(s_radiusmin, s_radiusmax)
"""
kwargs['radius_min'] = 15               # Minimum radius
kwargs['radius_max'] = 15              # Maximum radius
kwargs['edge_color'] = (1, 1, 1, 0.5)  # Color of the edges
# kwargs['edge_width'] = .5              # Width of the edges
kwargs['symbol'] = 'disc'            # Source's symbol

"""
Next, we mask source's data that are comprised between [20, 40] and color
each source to orange
"""
# # mask = np.logical_and(kwargs['data'] >= 20., kwargs['data'] <= 40)
# kwargs['mask'] = mask
# kwargs['mask_color'] = 'gray'

"""It's also possible to add text to each source. Here, we show the name of the
subject in yellow.
To avoid a superposition between the text and sources sphere, we introduce an
offset to the text using the s_textshift input
"""
# kwargs['text'] = subjects              # Name of the subject
# kwargs['text_color'] = "#f39c12"       # Set to yellow the text color
# kwargs['text_size'] = 1.5              # Size of the text
# kwargs['text_translate'] = (1.5, 1.5, 0)
# kwargs['text_bold'] = True

"""Create the source object. If you want to previsualize the result without
opening Brain, use s_obj.preview()
"""
s_obj = SourceObj('SourceExample', loc_mat, **kwargs)
vb = Brain(source_obj=s_obj, connect_obj=None)

vb.show()
# s_obj.preview()
# exit()

# """Color sources according to the data
# """
# # s_obj.color_sources(data=kwargs['data'], cmap='viridis')

# """Colorbar properties
# """
# cb_kw = dict(cblabel="Project source activity", cbtxtsz=3., border=False, )

# """Define a brain object with the B3 template and project source's activity
# onto the surface
# """
# b_obj = BrainObj('B1', **cb_kw)
# b_obj.project_sources(s_obj, cmap='viridis', vmin=50., under='orange',
#                       vmax=550., over='darkred')

# """Create a Brain instance and pass both of the brain and source object defined
# After the interface is opened, press C to display the colorbar.
# """
# vb = Brain(source_obj=s_obj, brain_obj=b_obj)
# vb.show()
