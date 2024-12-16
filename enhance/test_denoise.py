import torch
import random
from main import EnhanceDataset
from torchvision.transforms.v2.functional import to_dtype
import matplotlib.pyplot as plt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = torch.load('mprnet-30-epochs.pt').to(device).eval()
test_ds = EnhanceDataset("archive/train/train", "archive/train/gt")
test_samples = [test_ds[random.randint(0, len(test_ds) - 1)] for i in range(4)]

fig, ax = plt.subplots(nrows=4, ncols=3)
for i, (image, label) in enumerate(test_samples):
    image = to_dtype(image, torch.float32, scale=True)
    with torch.no_grad():
        pred, _, _ = model(image.unsqueeze(0).to(device))
        pred = pred.squeeze(0).cpu()
    ax[i, 0].imshow(image.permute(1, 2, 0))
    ax[i, 1].imshow(label.permute(1, 2, 0))
    ax[i, 2].imshow(pred.permute(1, 2, 0))
[a.set_axis_off() for a in ax.ravel()]
plt.tight_layout()
plt.show()
