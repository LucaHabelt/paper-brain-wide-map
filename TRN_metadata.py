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
Plot firing rate distribution of TRN units:
"""
area = "RT"

unit_df = bwm_units(one)
unit_df.describe()
br = BrainRegions()


area_id = br.id[br.acronym == area][0]
area_ids = br.descendants(area_id).id
area_df = unit_df[unit_df["atlas_id"].isin(area_ids)]



plt.figure()
plt.hist(area_df["firing_rate"], bins=50)
plt.xlabel("Firing rate (Hz)")
plt.ylabel("Number of units")
plt.title(f"{area} unit firing rate distribution")
plt.show()


print(f"{area} units:", len(area_df))
print("Median FR:", area_df["firing_rate"].median())
print("Mean FR:", area_df["firing_rate"].mean())





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
