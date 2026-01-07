import ssl
# Bypass certificate verification for this session
ssl._create_default_https_context = ssl._create_unverified_context
from brainbox.task.trials import get_event_aligned_raster, get_psth
from brainbox.behavior.wheel import velocity_filtered
import numpy as np
from one.api import ONE
from brainbox.io.one import SpikeSortingLoader, SessionLoader
from iblatlas.atlas import AllenAtlas
from brainwidemap import bwm_query, load_good_units, load_trials_and_mask, bwm_units
import matplotlib.pyplot as plt

ONE.setup(base_url='https://openalyx.internationalbrainlab.org', silent=True)
one = ONE(password='international')
ba = AllenAtlas()

"""
Connected to https://openalyx.internationalbrainlab.org as user "intbrainlab"
"""
bwm_df = bwm_query(one)
#print(bwm_df.to_markdown())


insertions_rt = one.search_insertions(atlas_acronym='RT', datasets='spikes.times.npy', project='brainwide')
# Display the found insertions
print(f"Found {len(insertions_rt)} insertions in RT")
print(insertions_rt)

pid = insertions_rt[0]
[eid, pname] = one.pid2eid(pid)

ssl = SpikeSortingLoader(pid=pid, one=one, atlas=ba)
spikes, clusters, channels = ssl.load_spike_sorting()
clusters = ssl.merge_clusters(spikes, clusters, channels)

good_cluster_idx = clusters['label'] == 1
good_cluster_IDs = clusters['cluster_id'][good_cluster_idx]

clusters_g = {key: val[good_cluster_idx] for key, val in clusters.items()}

good_spk_indx = np.where(np.isin(spikes['clusters'], good_cluster_IDs))
spikes_g = {key: val[good_spk_indx] for key, val in spikes.items()}

num_neuron = len(np.unique(spikes_g['clusters']))

# load trial data
sl = SessionLoader(eid=eid, one=one)
sl.load_trials()
events = sl.trials['firstMovement_times']

# If event == NaN, remove the trial from the analysis
nan_index = np.where(np.isnan(events))[0]
events = events.drop(index=nan_index).to_numpy()
contrast_R = sl.trials.contrastRight.drop(index=nan_index).to_numpy()
contrast_L = sl.trials.contrastLeft.drop(index=nan_index).to_numpy()
choice = sl.trials.choice.drop(index=nan_index).to_numpy()
block = sl.trials.probabilityLeft.drop(index=nan_index).to_numpy()

num_trial = len(events)
indx_choice_a = np.where(choice == -1)[0]
indx_choice_b = np.where(choice == 1)[0]

# ---------------------------------------------------
# Load wheel data
wheel = one.load_object(eid, 'wheel', collection='alf')
Fs = 1000
speed, _ = velocity_filtered(wheel.position, Fs)

print(f'N good unit: {num_neuron}, N trials: {num_trial}')

# ---------------------------------------------------
# Select units in a given brain region

# Remap the channel acronyms to the Beryl brain region parcellation (which is a higher order)
region_acr_ch = ba.regions.id2acronym(channels['atlas_id'], mapping='Beryl')
cluster_g_acr = region_acr_ch[clusters_g['channels']]
# Print each unit's brain region label
print(cluster_g_acr)

# Take the index of units in SCm
indx_rt = np.where(cluster_g_acr == 'RT')[0]
cluster_RT_IDs = good_cluster_IDs[indx_rt]
nunit = len(cluster_RT_IDs)


# ---------------------------------------------------
# Create PSTHs
binsize = 0.01  # bin size [sec] for neural binning
time_window = [-0.150, 0]

for count, clu_id in enumerate(cluster_RT_IDs):

    # Find spikes for this cluster
    spk_indx = np.where(np.isin(spikes_g['clusters'], clu_id))
    spikes_unit = {key: val[spk_indx] for key, val in spikes_g.items()}

    # Compute raster
    raster, timestamps = get_event_aligned_raster(spikes_unit['times'], events, tbin=binsize, values=None,
                                                  epoch=time_window, bin=True)
    # Compute PSTH (return only the mean)
    psth_a, _ = get_psth(raster, trial_ids=indx_choice_a)
    psth_b, _ = get_psth(raster, trial_ids=indx_choice_b)

    # ------- Stack PSTHs -------
    # Init ; Here we create a M = n condition x n unit x n time bin and will concatenate it later
    if count == 0:
        nbin = len(timestamps)
        stack_psth = np.empty((2, nunit, nbin))

    stack_psth[0, count, :] = psth_a
    stack_psth[1, count, :] = psth_b

# ---------------------------------------------------
# Plot stacked PSTHs
fig, ax = plt.subplots(2)
fig.set_size_inches(6.5, 5.5)
ax[0].imshow(stack_psth[0, :, :], vmax=stack_psth.max(), vmin=stack_psth.min())
ax[1].imshow(stack_psth[1, :, :], vmax=stack_psth.max(), vmin=stack_psth.min())
ax[0].set_xlabel('')
ax[1].set_xlabel('time bin')
ax[0].set_ylabel('PSTH CCW (right) choice')
ax[1].set_ylabel('PSTH CW (left) choice')

ytick_loc = [0]
ax[0].set_yticks(ytick_loc)
ax[1].set_yticks(ytick_loc)
ax[0].set_yticklabels(['unit #1'])
ax[1].set_yticklabels(['unit #1'])