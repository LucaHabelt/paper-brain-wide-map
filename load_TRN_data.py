import ssl
# Bypass certificate verification for this session
ssl._create_default_https_context = ssl._create_unverified_context

import numpy as np
from one.api import ONE
from brainbox.io.one import SpikeSortingLoader, SessionLoader
from iblatlas.atlas import AllenAtlas
from brainwidemap import bwm_query, load_good_units, load_trials_and_mask, bwm_units
from brainwidemap import load_good_units, load_trials_and_mask

ONE.setup(base_url='https://openalyx.internationalbrainlab.org', silent=True)
one = ONE(password='international')
ba = AllenAtlas()

"""
Connected to https://openalyx.internationalbrainlab.org as user "intbrainlab"
"""
bwm_df = bwm_query(one)

rt_cluster_indices = np.where(clusters['acronym'] == 'RT')[0]