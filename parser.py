import numpy as np
import torch
import zarr
from torch.utils.data import IterableDataset,DataLoader

DATASET="alpha_drone_mk2_stl_corrected/patient_one_data.zarr/"
OUT="data_sim/flight.npz"

class EpisodeDataset(IterableDataset):
    def __init__(self,path,episode=0,window=1):
        self.path=path
        self.episode=episode
        self.window=window
        self.length=zarr.open(zarr.storage.LocalStore(path),mode="r")["episodes"].shape[1]

    def __iter__(self):
        data=zarr.open(zarr.storage.LocalStore(self.path),mode="r")["episodes"]
        for i in range(self.length-self.window+1):
            yield torch.from_numpy(data[self.episode,i:i+self.window].astype(np.float32))

pos,quat,act=[],[],[]

for d in DataLoader(EpisodeDataset(DATASET),batch_size=None):
    s=d[0]
    pos.append(s[:3].numpy())
    quat.append(s[6:10].numpy())
    act.append((s[16:20].numpy())*8000/12.5)

np.savez(
    OUT,
    position=np.array(pos),
    quaternion=np.array(quat),
    motor=np.array(act)
)

print(f"saved {OUT}")
