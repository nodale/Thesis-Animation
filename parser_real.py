import numpy as np
import glob
import os
from pyulog import ULog
from scipy.spatial.transform import Rotation, Slerp
from scipy.interpolate import interp1d

DATA_DIR = "data_real"
HZ = 200

for log_dir in sorted(glob.glob(os.path.join(DATA_DIR, "log_*"))):
    ulg_files = glob.glob(os.path.join(log_dir, "*.ulg"))
    if not ulg_files:
        print(f"no .ulg in {log_dir}, skipping")
        continue

    ulg_path = ulg_files[0]
    log = ULog(ulg_path, ["vehicle_odometry", "johnny_status"])

    datasets = {d.name: d for d in log.data_list if d.multi_id == 0}

    odo = datasets["vehicle_odometry"].data
    js  = datasets["johnny_status"].data

    position   = np.stack([odo["position[0]"], odo["position[1]"], odo["position[2]"]], axis=1).astype(np.float64)
    quaternion = np.stack([odo["q[0]"],        odo["q[1]"],        odo["q[2]"],        odo["q[3]"]], axis=1).astype(np.float64)
    odo_ts     = odo["timestamp"].astype(np.float64)

    u     = np.stack([js["u[0]"], js["u[1]"], js["u[2]"], js["u[3]"]], axis=1).astype(np.float64)
    js_ts = js["timestamp"].astype(np.float64)

    # common time window covered by both topics (microseconds)
    t_start = max(odo_ts[0],  js_ts[0])
    t_end   = min(odo_ts[-1], js_ts[-1])
    n = int((t_end - t_start) / 1e6 * HZ)
    t_grid = np.linspace(t_start, t_start + (n - 1) * 1e6 / HZ, n)

    # position: linear interp
    pos_interp = interp1d(odo_ts, position, axis=0, assume_sorted=True)
    position_200 = pos_interp(t_grid).astype(np.float32)

    # quaternion: SLERP (scipy uses xyzw, ulog is wxyz)
    rots = Rotation.from_quat(quaternion[:, [1, 2, 3, 0]])
    slerp = Slerp(odo_ts, rots)
    quat_200 = slerp(t_grid).as_quat()[:, [3, 0, 1, 2]].astype(np.float32)

    # motor: linear interp + scale
    u_interp = interp1d(js_ts, u, axis=0, assume_sorted=True)
    motor_200 = (u_interp(t_grid) * 8000.0 / 12.5).astype(np.float32)

    # timestamp in seconds from start
    ts_200 = ((t_grid - t_start) / 1e6).astype(np.float32)

    out = os.path.join(log_dir, "flight.npz")
    np.savez(
        out,
        timestamp=ts_200,
        position=position_200,
        quaternion=quat_200,
        motor=motor_200,
    )
    print(f"saved {out}  |  {n} samples @ {HZ} Hz  ({ts_200[-1]:.1f} s)")
