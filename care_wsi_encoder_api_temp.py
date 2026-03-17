
import torch
import os
import numpy as np
from models.CARE import CareSSL as CARE

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
model = CARE(embed_dim=512, id_dim = 768, num_heads=8, num_region = 8)
ckpt = torch.load("models/CARE.pt", map_location=torch.device('cpu'), weights_only=True)

msg = model.load_state_dict(ckpt,strict=False)
#print(msg)

model.to(device)
model = model.eval()
N_values = torch.tensor([coords.shape[1]], dtype=torch.long, device=coords.device)
#get wsi enbedding
with torch.inference_mode():
    wsi_embedding, AR_embedding,num_AR,AR_loss = model(patch_embedding,N_values, coords, return_wsi = True)     
print(wsi_embedding.shape, AR_embedding.shape,num_AR,AR_loss)
