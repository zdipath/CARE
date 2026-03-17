
import torch
import os
import numpy as np
from transformers import AutoModel
device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
fea = np.load('./data/MUT/conch_v1_5/19579_0_1024.npy', allow_pickle=True)
features = fea[()]['feature']
cor = fea[()]['index']
coords = np.array([filename.split('_')[:2] for filename in cor], dtype=int)
patch_embedding = torch.from_numpy(features).unsqueeze(0).to(device)
coords = torch.from_numpy(coords).unsqueeze(0)
N_values = torch.zeros((1)).to(device)
N_values[0] = coords.shape[1]
for i in range(coords.shape[0]):
    diffs = np.linalg.norm(coords[i][1:] - coords[i][:-1], axis=1)
    count_512 = np.sum(diffs == 512)
    count_1024 = np.sum(diffs == 1024)
    count_256  = np.sum(diffs == 256)
    count_128  = np.sum(diffs == 128)
    counts = {
        256: count_256,
        128: count_128,
        1024: count_1024,
        512: count_512,
    }
    
    patch_size = max(counts, key=counts.get)
    coords[i] = coords[i] // patch_size
coords = coords.to(device)
model_base = AutoModel.from_pretrained("Zipper-1/CARE",trust_remote_code=True)
model_base.to(device)
model_base = model_base.eval()
with torch.inference_mode():
    out = model_base(patch_embedding,N_values, coords)     
print(out)
print(out.wsi_embedding[0,:10])
print(out.aux_loss)
