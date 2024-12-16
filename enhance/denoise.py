import os
import torch
from tqdm import tqdm
from torchvision.io import read_image
from torchvision.transforms.v2.functional import to_pil_image, to_dtype

op = os.path
PATHS = [
    "archive/train/train",
    "archive/val/val",
    "archive/test/",
]
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = torch.load("mprnet-30-epochs.pt")
for path in PATHS:
    files = [op.join(path, f) for f in os.listdir(path)]
    for file in tqdm(files):
        outpath = file.replace('archive/', 'denoised/')
        image = read_image(file)
        image = to_dtype(image, torch.float32, scale=True)
        with torch.no_grad():
            denoised, _, _ = model(image.unsqueeze(0).to(device))
        denoised = to_pil_image(denoised.squeeze(0), mode='RGB')
        denoised.save(outpath)
