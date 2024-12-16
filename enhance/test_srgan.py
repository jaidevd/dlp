import torch
from torchvision.io import read_image
from torchvision.transforms.v2.functional import to_dtype, to_pil_image
from srgan import Generator
import os
from tqdm import tqdm

op = os.path
model = Generator()
model.load_state_dict(torch.load('generator-best.pth', weights_only=True))
model.eval()

root = "denoised/test/"
for file in tqdm(os.listdir(root)):
    img = read_image(op.join(root, file))
    img = to_dtype(img, torch.float32, scale=True)
    with torch.no_grad():
        out = model(img.unsqueeze(0))
    out = out.squeeze(0)
    to_pil_image(out).save(op.join("/tmp/out", file))
