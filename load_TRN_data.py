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

one = ONE(base_url="https://openalyx.internationalbrainlab.org")
bwm_df = bwm_query(one)

insertions_rt = one.search_insertions(atlas_acronym='RT', datasets='spikes.times.npy', project='brainwide')


"""
Plot firing rate distribution of TRN units:
"""


unit_df = bwm_units(one)
unit_df.describe()

br = BrainRegions()
rt_id = br.id[br.acronym == "TRN"][0]
rt_ids = br.descendants(rt_id).id
rt_df = unit_df[unit_df["atlas_id"].isin(rt_ids)]



plt.figure()
plt.hist(rt_df["firing_rate"], bins=50)
plt.xlabel("Firing rate (Hz)")
plt.ylabel("Number of units")
plt.title("TRN unit firing rate distribution")
plt.show()


print("RT units:", len(rt_df))
print("Median FR:", rt_df["firing_rate"].median())
print("Mean FR:", rt_df["firing_rate"].mean())



"""
PSTH for TRN units relative to task epochs
"""

ba = AllenAtlas()

#for i in range(17,25):

for pid in tqdm(insertions_rt[:25]):

    #pid = insertions_rt[i]
    #pid = 'af775340-53d0-43cd-9cda-48e32b95adca'

    # ---------- load spikes / clusters / channels ----------
    ssl = SpikeSortingLoader(pid=pid, one=one, atlas=ba)
    spikes, clusters, channels = ssl.load_spike_sorting()
    clusters = ssl.merge_clusters(spikes, clusters, channels)

    good = clusters["label"] == 1

    ch_acr = ba.regions.id2acronym(channels["atlas_id"], mapping="Allen")
    cluster_acr = ch_acr[clusters["channels"]]

    rt_unit_ids = clusters["cluster_id"][good & (cluster_acr == "RT")]
    print("RT good units on this probe:", len(rt_unit_ids))

    # ---------- load trial events ----------
    eid, _ = one.pid2eid(pid)
    sl = SessionLoader(eid=eid, one=one)
    sl.load_trials()

    # pick an epoch/event to align to (choose one that exists in your trials table)
    event_name = "firstMovement_times"   # try also: "firstMovement_times", "stimOn_times", "feedback_times"
    events = sl.trials[event_name].to_numpy()
    events = events[~np.isnan(events)]

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


    # ---------- all RT units on this probe (heatmap + mean) ----------
    all_rates = []
    t = None
    for uid in rt_unit_ids:
        t, r = aligned_counts(get_unit_spike_times(int(uid)), events, t_before=3, t_after=1.5, bin_size=0.1)
        all_rates.append(r.mean(axis=0))

    all_rates = np.vstack(all_rates)

    plt.figure()
    plt.imshow(all_rates, aspect="auto", origin="lower",
               extent=[t[0], t[-1], 0, all_rates.shape[0]])
    plt.axvline(0, linestyle="--")
    plt.xlabel(f"Time from {event_name} (s)")
    plt.ylabel("RT units (index)")
    plt.title(f"PID {pid} | RT units PSTH heatmap")
    plt.colorbar(label="Hz")
    #plt.show()

    #plt.savefig(fr"Z:\home\lh1192\Backup_2025_11_03\PhD\TRN\IBL_BWM_plots\PSTH_stimulusonset_singleprobe\TRN_PSTH_heatmap_{pid}.svg")
    #plt.close()

    plt.figure()
    for r in all_rates:
        plt.plot(t, r, color="gray", alpha=0.3)

    plt.plot(t, all_rates.mean(axis=0), color="black", linewidth=2)
    plt.axvline(0, linestyle="--")
    plt.xlabel(f"Time from {event_name} (s)")
    plt.ylabel("Mean firing rate (Hz)")
    plt.title(f"PID {pid} | RT population mean PSTH")
    #plt.show()

    #plt.savefig(fr"Z:\home\lh1192\Backup_2025_11_03\PhD\TRN\IBL_BWM_plots\PSTH_stimulusonset_singleprobe\TRN_population_mean_PSTH_{pid}.svg")
    #plt.close()




"""
Visualization of Stimulus movement onset time interval 
"""


rows = []

for pid in tqdm(insertions_rt[:25]):
    eid, _ = one.pid2eid(pid)

    sl = SessionLoader(eid=eid, one=one)
    sl.load_trials()
    tr = sl.trials

    stim = tr["stimOn_times"].to_numpy()
    move = tr["firstMovement_times"].to_numpy()

    ok = (~np.isnan(stim)) & (~np.isnan(move))
    dt = move[ok] - stim[ok]

    # optional: keep only movements after stim
    dt = dt[dt >= 0]

    rows.append({
        "pid": str(pid),
        "eid": str(eid),
        "n_trials": int(dt.size),
        "dt_mean_s": float(np.mean(dt)) if dt.size else np.nan,
        "dt_sd_s": float(np.std(dt, ddof=1)) if dt.size > 1 else np.nan,
        "dt_median_s": float(np.median(dt)) if dt.size else np.nan,
    })

dt_df = pd.DataFrame(rows)
dt_df


plt.figure()
plt.hist(dt_df["dt_mean_s"].dropna(), bins=50)
plt.xlabel("Mean (firstMovement - stimOn) per probe (s)")
plt.ylabel("Number of probes")
plt.title("Movement latency relative to stimulus (per probe)")
plt.show()



x = np.arange(len(dt_df))

plt.figure(figsize=(10, 4))

plt.errorbar(
    x,
    dt_df["dt_mean_s"],
    yerr=dt_df["dt_sd_s"],
    fmt="o",
    capsize=4
)

for i, n in enumerate(dt_df["n_trials"]):
    plt.text(i, dt_df["dt_mean_s"].iloc[i], f"n={n}",
             ha="center", va="bottom", fontsize=8)

plt.xticks(x, [pid[:8] for pid in dt_df["pid"]], rotation=45)
plt.ylabel("Movement − stimulus latency (s)")
plt.xlabel("Probe (pid)")
plt.title("Reaction time per probe (mean ± SD)")

plt.tight_layout()
plt.show()



"""
coordinates od TRN units
"""


def mean_rt_xyz_for_probe(pid):
ssl = SpikeSortingLoader(pid=pid, one=one, atlas=ba)
spikes, clusters, channels = ssl.load_spike_sorting()
clusters = ssl.merge_clusters(spikes, clusters, channels)

good = clusters["label"] == 1

ch_acr = ba.regions.id2acronym(channels["atlas_id"], mapping="Allen")
cluster_acr = ch_acr[clusters["channels"]]

rt_mask = good & (cluster_acr == "RT")
if not np.any(rt_mask):
    return None

peak_ch = clusters["channels"][rt_mask].astype(int)

x = channels["x"][peak_ch]
y = channels["y"][peak_ch]
z = channels["z"][peak_ch]

if np.nanmax(np.abs(x)) < 1:
    x, y, z = x * 1e6, y * 1e6, z * 1e6

return {
    "pid": pid,
    "n_rt_units": rt_mask.sum(),
    "x_um": np.mean(x),
    "y_um": np.mean(y),
    "z_um": np.mean(z),
}


rows = []
for pid in insertions_rt:
r = mean_rt_xyz_for_probe(pid)
if r is not None:
    rows.append(r)

rt_probe_xyz = pd.DataFrame(rows)

mtrn_probe_xyz = rt_probe_xyz[rt_probe_xyz["y_um"] >-800]
