import numpy as np
import pandas as pd
from one.api import ONE
from brainbox.io.one import SpikeSortingLoader
from iblatlas.atlas import Insertion, ALLEN_CCF_LANDMARKS_MLAPDV_UM
from brainrender import Scene
from brainrender.actors import Line
from iblatlas.regions import BrainRegions
from brainwidemap import bwm_query, load_good_units, load_trials_and_mask, bwm_units
from iblatlas.atlas import AllenAtlas
from brainbox.io.one import SpikeSortingLoader, SessionLoader
import matplotlib.pyplot as plt
from tqdm import tqdm


# ---------- PSTH helpers ----------
def aligned_counts(spike_times, event_times, t_before=0.5, t_after=1.0, bin_size=0.01):
    edges = np.arange(-t_before, t_after + bin_size, bin_size)
    counts = np.zeros((event_times.size, edges.size - 1), dtype=float)
    for i, t0 in enumerate(event_times):
        rel = spike_times[(spike_times >= t0 - t_before) & (spike_times <= t0 + t_after)] - t0
        counts[i], _ = np.histogram(rel, bins=edges)
    rate = counts / bin_size
    t = (edges[:-1] + edges[1:]) / 2
    return t, rate


def get_unit_spike_times(unit_id):
    return spikes["times"][spikes["clusters"] == unit_id]


"""
Loading data from  IBl server
"""
one = ONE(base_url="https://openalyx.internationalbrainlab.org")
bwm_df = bwm_query(one)
ba = AllenAtlas()


"""
sorting probe IDs recording from target area
"""
insertions_rt = one.search_insertions(atlas_acronym='RT', datasets='spikes.times.npy', project='brainwide')
insertions_val = one.search_insertions(atlas_acronym='VAL', datasets='spikes.times.npy', project='brainwide')
insertions_vm = one.search_insertions(atlas_acronym='VM', datasets='spikes.times.npy', project='brainwide')
insertions_vpl = one.search_insertions(atlas_acronym='VPL', datasets='spikes.times.npy', project='brainwide')



"""
PSTH for target area units relative to task epochs
"""

area = "VPL"

for pid in tqdm(insertions_vpl[4:26]):
#pid = 'ca073754-be17-43b7-a38a-0c1e5563ff32'

    # ---------- load spikes / clusters / channels ----------
    ssl = SpikeSortingLoader(pid=pid, one=one, atlas=ba)
    spikes, clusters, channels = ssl.load_spike_sorting()
    clusters = ssl.merge_clusters(spikes, clusters, channels)

    good = clusters["label"] == 1

    ch_acr = ba.regions.id2acronym(channels["atlas_id"], mapping="Allen")
    cluster_acr = ch_acr[clusters["channels"]]

    area_unit_ids = clusters["cluster_id"][good & (cluster_acr == area)]
    print(f"{area} good units on this probe:", len(area_unit_ids))

    # ---------- load trial events ----------
    eid, _ = one.pid2eid(pid)
    sl = SessionLoader(eid=eid, one=one)
    sl.load_trials()

    # pick an epoch/event to align to (choose one that exists in your trials table)
    event_name = "firstMovement_times"   # try also: "firstMovement_times", "stimOn_times", "feedback_times"
    events = sl.trials[event_name].to_numpy()
    events = events[~np.isnan(events)]



    # ---------- all RT units on this probe (heatmap + mean) ----------
    all_rates = []
    t = None
    for uid in area_unit_ids:
        t, r = aligned_counts(get_unit_spike_times(int(uid)), events, t_before=3, t_after=1.5, bin_size=0.1)
        all_rates.append(r.mean(axis=0))

    all_rates = np.vstack(all_rates)

    plt.figure()
    plt.imshow(all_rates, aspect="auto", origin="lower",
               extent=[t[0], t[-1], 0, all_rates.shape[0]])
    plt.axvline(0, linestyle="--")
    plt.xlabel(f"Time from {event_name} (s)")
    plt.ylabel(f"{area} units (index)")
    plt.title(f"PID {pid} | {area} units PSTH heatmap")
    plt.colorbar(label="Hz")
    #plt.show()

    plt.savefig(fr"Z:\home\lh1192\Backup_2025_11_03\PhD\TRN\IBL_BWM_plots\PSTH_moveonset_singleprobe_v02\{area}\{area}_PSTH_heatmap_{pid}.svg")
    plt.close()

    plt.figure()
    for r in all_rates:
        plt.plot(t, r, color="gray", alpha=0.3)

    plt.plot(t, all_rates.mean(axis=0), color="black", linewidth=2)
    plt.axvline(0, linestyle="--")
    plt.xlabel(f"Time from {event_name} (s)")
    plt.ylabel("Mean firing rate (Hz)")
    plt.title(f"PID {pid} | {area} population mean PSTH")
    #plt.show()

    plt.savefig(fr"Z:\home\lh1192\Backup_2025_11_03\PhD\TRN\IBL_BWM_plots\PSTH_moveonset_singleprobe_v02\{area}\{area}_population_mean_PSTH_{pid}.svg")
    plt.close()



