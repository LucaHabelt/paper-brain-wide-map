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
from tqdm import tqdm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


""" z-score helpers """

def per_trial_baseline_windows(stim, move, fb, guard=0.2, min_len=0.2):
    starts, ends = [], []
    n = len(stim)
    for i in range(n):
        if i == 0:
            continue
        if np.isnan(fb[i-1]) or np.isnan(stim[i]):
            continue
        t_start = fb[i-1] + guard
        t_end = stim[i] - guard
        if not np.isnan(move[i]):
            t_end = min(t_end, move[i] - guard)
        if t_end - t_start >= min_len:
            starts.append(t_start)
            ends.append(t_end)
    return np.array(starts), np.array(ends)

def baseline_mean_sd(spike_times, starts, ends):
    if starts.size == 0:
        return np.nan, np.nan
    rates = []
    for t0, t1 in zip(starts, ends):
        nspk = np.sum((spike_times >= t0) & (spike_times < t1))
        rates.append(nspk / (t1 - t0))
    rates = np.asarray(rates)
    return rates.mean(), rates.std(ddof=1) if rates.size > 1 else (rates.mean(), 0.0)

def psth_rate(spike_times, event_times, t_before=3, t_after=1.5, bin_size=0.1):
    edges = np.arange(-t_before, t_after + bin_size, bin_size)
    t = (edges[:-1] + edges[1:]) / 2
    counts = np.zeros((event_times.size, edges.size - 1), float)
    for i, t0 in enumerate(event_times):
        rel = spike_times[(spike_times >= t0 - t_before) & (spike_times <= t0 + t_after)] - t0
        counts[i], _ = np.histogram(rel, bins=edges)
    return t, (counts / bin_size).mean(axis=0)  # mean across trials

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
insertions_gpe = one.search_insertions(atlas_acronym='GPe', datasets='spikes.times.npy', project='brainwide')
insertions_gpi = one.search_insertions(atlas_acronym='GPi', datasets='spikes.times.npy', project='brainwide')
insertions_zi = one.search_insertions(atlas_acronym='ZI', datasets='spikes.times.npy', project='brainwide')
insertions_mo = one.search_insertions(atlas_acronym='MO', datasets='spikes.times.npy', project='brainwide')
insertions_MOp = one.search_insertions(atlas_acronym='MOp', datasets='spikes.times.npy', project='brainwide')
insertions_MOs = one.search_insertions(atlas_acronym='MOs', datasets='spikes.times.npy', project='brainwide')

areas = ["MOp", "MOs"]
insertions_by_area = {
    "MOp": insertions_MOp,
    "MOs": insertions_MOs,
}

for area in areas:

    use_bootstrap_95ci = True

    Z_all = []
    t_ref = None
    event_name = "firstMovement_times"



    for pid in tqdm(insertions_by_area[area],desc=f"insertions_{area}"):


        """ helpers """

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


        def bootstrap_ci_units(Y, n_boot=1000, alpha=0.05, seed=1):
            rng = np.random.default_rng(seed)
            n = Y.shape[0]
            idx = rng.integers(0, n, size=(n_boot, n))
            boot_means = Y[idx].mean(axis=1)
            lo = np.quantile(boot_means, alpha / 2, axis=0)
            hi = np.quantile(boot_means, 1 - alpha / 2, axis=0)
            return lo, hi


        """ load spike data """
        ssl = SpikeSortingLoader(pid=pid, one=one, atlas=ba)
        spikes, clusters, channels = ssl.load_spike_sorting()
        clusters = ssl.merge_clusters(spikes, clusters, channels)
        good = clusters["label"] == 1

        ch_acr = ba.regions.id2acronym(channels["atlas_id"], mapping="Allen")
        cluster_acr = ch_acr[clusters["channels"]]

        area_unit_ids = clusters["cluster_id"][good & (cluster_acr == area)]
        print(f"{area} good units on this probe:", len(area_unit_ids))
        if len(area_unit_ids) == 0:
            continue

        """ load trial events """
        eid, _ = one.pid2eid(pid)
        sl = SessionLoader(eid=eid, one=one)
        sl.load_trials()

        event_name = "firstMovement_times"  # choose eent from: "firstMovement_times", "stimOn_times", "feedback_times"
        events = sl.trials[event_name].to_numpy()
        events = events[~np.isnan(events)]
        if events.size == 0:
            continue

        """trial specific baseline window for z-scoring"""
        stim = sl.trials["stimOn_times"].to_numpy()
        move = sl.trials["firstMovement_times"].to_numpy()
        fb = sl.trials["feedback_times"].to_numpy()
        b_starts, b_ends = per_trial_baseline_windows(stim, move, fb, guard=0.2, min_len=0.2)
        if b_starts.size == 0:
            continue

        """ all units on this probe """
        all_rates = []
        t = None
        for uid in area_unit_ids:
            st = get_unit_spike_times(int(uid))


            mu, sd = baseline_mean_sd(st, b_starts, b_ends) # baseline mean/sd per unit
            sd = sd + 1e-9

            t, r = aligned_counts(st, events, t_before=3, t_after=1.5, bin_size=0.1)
            r_mean = r.mean(axis=0)


            r_mean = (r_mean - mu) / sd # z-score the PSTH using baseline

            all_rates.append(r_mean)

        all_rates = np.vstack(all_rates)

        if t_ref is None:
            t_ref = t
        else:
            if len(t) != len(t_ref) or np.max(np.abs(t - t_ref)) > 1e-12:
                raise RuntimeError("Time axis changed across probes.")

        Z_all.append(all_rates)  # all_rates already z-scored per unit

    """Plot bootstrapped across units in specific area"""
    all_rates_z = np.vstack(Z_all)   # (all units across all probes, n_bins)
    m = all_rates_z.mean(axis=0)
    n_units = all_rates_z.shape[0]

    plt.figure()
    plt.plot(t_ref, m, color="black", linewidth=2)

    if use_bootstrap_95ci and n_units >= 2:
        lo, hi = bootstrap_ci_units(all_rates_z, n_boot=1000, seed=1)
        plt.fill_between(t_ref, lo, hi, color="black", alpha=0.3)
    else:
        if n_units >= 2:
            sem = all_rates_z.std(axis=0, ddof=1) / np.sqrt(n_units)
            plt.fill_between(t_ref, m - sem, m + sem, color="black", alpha=0.3)

    plt.axvline(0, linestyle="--")
    plt.xlabel(f"Time from {event_name} (s)")
    plt.ylabel("Mean (z)")
    plt.title(f"{area} | ALL probes pooled | n={n_units} units")
    plt.savefig(
            fr"Z:\home\lh1192\Backup_2025_11_03\PhD\TRN\IBL_BWM_plots\PSTH_moveonset_zstack_bootstrap\{area}_PSTH_zstack_bootstrap.svg")
