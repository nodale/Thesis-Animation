import numpy as np
import glob
import os
from pyulog import ULog
from scipy.spatial.transform import Rotation, Slerp
from scipy.interpolate import interp1d
import zarr

DATA_DIR = "data_inference"
HZ = 30
POSITION_SCALE = 3.0      # norm.pos from the collection pipeline's hydra config
REANCHOR_INTERVAL = 32.0  # seconds; matches nn_estimate.reseed_every_s in that config

# --- Timing alignment ---------------------------------------------------------------
#
# Ground truth position/orientation comes straight from the zarr's own "ground_truth"/
# "obs" columns (already correctly time-synced with "output", the inferred deltas), per
# data_inference/plot_inference_sideways.py -- no .ulg cross-referencing needed for that.
#
# But the zarr's own logging does NOT start at the same moment the .ulg recording does:
# the online-learning script is launched manually, partway into an already-flying drone,
# so there's a real, non-zero delay between .ulg-start and the zarr's first logged row.
# That delay has to be measured empirically (there's no field that records it directly) --
# cross-correlate a motion signal common to both sides (the zarr's own ground truth vs.
# the same quantity computed from the .ulg) to find it. This also gives the offset needed
# to place motor commands (from .ulg johnny_status, not present in the zarr) onto the
# zarr's own time axis.

def estimate_start_delay(t_s, ground_truth_speed_v, ground_truth_speed_t, odo_ts, gt_speed, gt_speed_t,
                          search_s=80.0, win_s=25.0, hop_s=12.5, grid_hz=50.0, n_windows_used=6):
    """Cross-correlate a short window near the start of the zarr recording against the
    .ulg's own motion signal to find delay_s such that:
        ulog_time_us(row) = odo_ts[0] + (t_s[row] - t_s[0])*1e6 + delay_s*1e6
    """
    gt_grid_t = np.arange(gt_speed_t[0], gt_speed_t[-1], 1e6 / grid_hz)
    gt_v = interp1d(gt_speed_t, gt_speed, bounds_error=False, fill_value=0)(gt_grid_t)
    gt_v = gt_v - gt_v.mean()

    obs_t = ground_truth_speed_t  # elapsed seconds since t_s[0]
    n_windows = min(n_windows_used, int((obs_t[-1] - win_s) // hop_s))
    delays, scores = [], []
    for w in range(max(n_windows, 1)):
        c = obs_t[0] + w * hop_s + win_s / 2
        mask = (obs_t > c - win_s / 2) & (obs_t < c + win_s / 2)
        if mask.sum() < 5:
            continue
        seg_t, seg_v = obs_t[mask], ground_truth_speed_v[mask]
        seg_grid_t = np.arange(seg_t[0], seg_t[-1], 1 / grid_hz)
        if len(seg_grid_t) < 5:
            continue
        seg_v = interp1d(seg_t, seg_v, bounds_error=False, fill_value=0)(seg_grid_t)
        seg_v = seg_v - seg_v.mean()

        guess_center_ulog = odo_ts[0] + c * 1e6
        lo, hi = guess_center_ulog - search_s * 1e6, guess_center_ulog + search_s * 1e6
        cand_starts = np.arange(lo, hi, 1e6 / grid_hz)
        best_score, best_lag = -np.inf, None
        for cs in cand_starts:
            idx0 = int((cs - gt_grid_t[0]) / (1e6 / grid_hz))
            if idx0 < 0 or idx0 + len(seg_v) > len(gt_v):
                continue
            score = np.dot(gt_v[idx0:idx0 + len(seg_v)], seg_v)
            if score > best_score:
                best_score, best_lag = score, cs
        if best_lag is None:
            continue
        matched_center_ulog = best_lag + (win_s / 2) * 1e6
        delays.append((matched_center_ulog - guess_center_ulog) / 1e6)
        scores.append(best_score)

    if not delays:
        raise RuntimeError("could not estimate start delay -- no correlated windows found")
    delays, scores = np.array(delays), np.array(scores)
    return float(np.median(delays[scores > np.median(scores) * 0.5])), delays, scores


def find_laps(t_s, lap_duration=32.0, min_lap_duration=16.0):
    """Ported from plot_inference_sideways.py: split a (single-generation) time
    array into fixed-duration laps, merging a too-short trailing lap into the
    previous one. Returns a list of (start_idx, end_idx) index slices."""
    t0 = t_s[0]
    laps = []
    lap_start = 0
    lap_end_time = t0 + lap_duration
    for i in range(1, len(t_s)):
        if t_s[i] >= lap_end_time:
            laps.append((lap_start, i))
            lap_start = i
            lap_end_time = t_s[i] + lap_duration
    if lap_start < len(t_s) - 1:
        laps.append((lap_start, len(t_s)))
    laps = laps if laps else [(0, len(t_s))]

    if len(laps) >= 2:
        last_i0, last_i1 = laps[-1]
        if t_s[last_i1 - 1] - t_s[last_i0] < min_lap_duration:
            prev_i0, _ = laps[-2]
            laps = laps[:-2] + [(prev_i0, last_i1)]
    return laps


for log_dir in sorted(glob.glob(os.path.join(DATA_DIR, "*"))):
    if not os.path.isdir(log_dir):
        continue

    ulg_files = glob.glob(os.path.join(log_dir, "*.ulg"))
    zarr_path = os.path.join(log_dir, "infer_log.zarr")
    if not ulg_files or not os.path.isdir(zarr_path):
        print(f"missing .ulg or infer_log.zarr in {log_dir}, skipping")
        continue

    is_sim = os.path.basename(log_dir).startswith("sim_")
    # "sim_" datasets' ground truth doesn't correlate with vehicle_odometry at all (they're
    # genuine simulated rollouts, not replays of that specific flight) -- confirmed
    # empirically; vehicle_visual_odometry does correlate for those instead.
    delay_topic = "vehicle_visual_odometry" if is_sim else "vehicle_odometry"

    ulg_path = ulg_files[0]
    log = ULog(ulg_path, ["johnny_status", delay_topic])
    datasets = {d.name: d for d in log.data_list if d.multi_id == 0}
    js = datasets["johnny_status"].data
    u     = np.stack([js["u[0]"], js["u[1]"], js["u[2]"], js["u[3]"]], axis=1).astype(np.float64)
    js_ts = js["timestamp"].astype(np.float64)  # us, ULog-internal clock

    odo = datasets[delay_topic].data
    odo_ts = odo["timestamp"].astype(np.float64)

    z = zarr.open(zarr.storage.LocalStore(zarr_path), mode="r")
    output = z["output"][:, 0:6].astype(np.float64)  # first 6 dims are consistent across model variants
    deltas = output[:, 0:3] * POSITION_SCALE

    obs = z["obs"][:].astype(np.float64)
    ground_truth = z["ground_truth"][:].astype(np.float64)
    generation_id = z["generation_id"][:].reshape(-1)
    position_gt   = ground_truth[:, 0:3] * POSITION_SCALE  # zarr's own ground truth, row-native time
    quaternion_gt = obs[:, 6:10]                            # zarr's own ground truth quaternion (w,x,y,z)

    raw_t = z["state_arrival_ns"][:]
    if np.asarray(raw_t).size == 0:
        raw_t = z["timestamp_ns"][:]
    t_s = np.asarray(raw_t).reshape(-1).astype(np.float64) / 1e9  # zarr's own row-native wall-clock time
    elapsed_zarr_s = t_s - t_s[0]

    if is_sim:
        # translational motion correlates far better than rotation for these datasets
        pos_gt_ulg = np.stack([odo["position[0]"], odo["position[1]"], odo["position[2]"]], axis=1).astype(np.float64)
        gt_speed = np.linalg.norm(np.diff(pos_gt_ulg, axis=0), axis=1) / (np.diff(odo_ts) / 1e6)
        gt_speed_t = odo_ts[1:]
        obs_speed = np.linalg.norm(np.diff(position_gt, axis=0), axis=1) / np.diff(elapsed_zarr_s)
        obs_speed_t = elapsed_zarr_s[1:]
    else:
        quat_gt_ulg = np.stack([odo["q[0]"], odo["q[1]"], odo["q[2]"], odo["q[3]"]], axis=1).astype(np.float64)
        rots_gt = Rotation.from_quat(quat_gt_ulg[:, [1, 2, 3, 0]])
        rel_gt = rots_gt[1:] * rots_gt[:-1].inv()
        gt_speed = rel_gt.magnitude() / (np.diff(odo_ts) / 1e6)
        gt_speed_t = odo_ts[1:]
        rots_obs = Rotation.from_quat(quaternion_gt[:, [1, 2, 3, 0]])
        rel_obs = rots_obs[1:] * rots_obs[:-1].inv()
        obs_speed = rel_obs.magnitude() / np.diff(elapsed_zarr_s)
        obs_speed_t = elapsed_zarr_s[1:]

    delay_s, delays, delay_scores = estimate_start_delay(t_s, obs_speed, obs_speed_t, odo_ts, gt_speed, gt_speed_t)
    # offset such that ulog_time_us(row) = (t_s[row] - offset) * 1e6
    offset = t_s[0] - odo_ts[0] / 1e6 - delay_s
    print(f"[{log_dir}] start delay={delay_s:.3f}s (n={len(delays)} windows, "
          f"spread={np.std(delays):.3f}s), offset={offset:.4f}s")

    # motor commands aren't in the zarr; place them onto the zarr's own row-native time
    # axis so they can be resampled alongside position/quaternion.
    js_ts_zarr_time = js_ts / 1e6 + offset  # seconds, on the zarr's own t_s scale

    # inferred position is dead-reckoned (integrated deltas) and drifts from the true
    # trajectory over time; re-anchor it to the ground-truth position at the start of
    # every generation (model swap) and every REANCHOR_INTERVAL seconds within a
    # generation -- exactly matching plot_inference_sideways.py's find_laps()+infer_xyz()
    # (generation boundaries are event-driven, not aligned to any global clock, so a
    # simple continuous floor(elapsed_s/32) reanchor lands at the wrong points entirely).
    position = np.zeros_like(deltas)
    for cid in np.unique(generation_id):
        idx = np.flatnonzero(generation_id == cid)
        for i0_rel, i1_rel in find_laps(t_s[idx], lap_duration=REANCHOR_INTERVAL):
            i0, i1 = idx[i0_rel], idx[i1_rel - 1] + 1
            position[i0] = position_gt[i0]
            if i1 - i0 > 1:
                position[i0 + 1:i1] = position_gt[i0] + np.cumsum(deltas[i0:i1 - 1], axis=0)

    t_start = max(t_s[0], js_ts_zarr_time[0])
    t_end   = min(t_s[-1], js_ts_zarr_time[-1])
    if t_end <= t_start:
        print(f"no overlap between zarr and motor timestamps in {log_dir}, skipping")
        continue
    n = int((t_end - t_start) * HZ)
    t_grid = np.linspace(t_start, t_start + (n - 1) / HZ, n)  # seconds, zarr time scale

    # position: linear interp (zarr's own time is already strictly increasing per row)
    pos_interp = interp1d(t_s, position, axis=0, assume_sorted=True)
    position_out = pos_interp(t_grid).astype(np.float32)

    # quaternion: zarr's own ground truth, SLERP (scipy uses xyzw, this is wxyz)
    rots = Rotation.from_quat(quaternion_gt[:, [1, 2, 3, 0]])
    slerp = Slerp(t_s, rots)
    quat_out = slerp(t_grid).as_quat()[:, [3, 0, 1, 2]].astype(np.float32)

    # motor: linear interp + scale
    u_interp = interp1d(js_ts_zarr_time, u, axis=0, assume_sorted=True)
    motor_out = (u_interp(t_grid) * 8000.0 / 12.5).astype(np.float32)

    # timestamp in seconds from start
    ts_out = (t_grid - t_start).astype(np.float32)

    # absolute ULog-internal time that timestamp=0 maps to (t_start is on the zarr's own
    # wall-clock scale; subtracting the same offset used for the motor alignment converts
    # it back to ULog-internal us, directly comparable to parser_real.py's own t_start_us)
    t_start_us = (t_start - offset) * 1e6

    out = os.path.join(log_dir, "flight_infer.npz")
    np.savez(
        out,
        timestamp=ts_out,
        position=position_out,
        quaternion=quat_out,
        motor=motor_out,
        t_start_us=np.float64(t_start_us),
    )
    print(f"saved {out}  |  {n} samples @ {HZ} Hz  ({ts_out[-1]:.1f} s)")
