import fcos
import torch
from functools import partial
from tqdm import tqdm

root = 'data/train'
dftrain, dftest = fcos.make_df(root, stratify=True)
dstrain, dstest = fcos.TrainDataset(dftrain, root), fcos.TrainDataset(dftest, root)
train_loader = torch.utils.data.DataLoader(
    dstrain, batch_size=256, shuffle=True,  # num_workers=6,
    collate_fn=partial(fcos.collate, device=False, resize=True),
)
test_loader = torch.utils.data.DataLoader(
    dstest, batch_size=128, shuffle=False,  # num_workers=2,
    collate_fn=partial(fcos.collate, device=False, resize=True)

)
for i, (images, targets) in tqdm(enumerate(train_loader), total=len(train_loader)):
    torch.save(images, f"precomp/train/images-batch-{i:02d}.pt")
    labels = torch.stack([t['labels'][0] for t in targets])
    torch.save(labels, f"precomp/train/labels-batch-{i:02d}.pt")
    boxes = torch.stack([t['boxes'][0] for t in targets])
    torch.save(boxes, f"precomp/train/boxes-batch-{i:02d}.pt")


for i, (images, targets) in tqdm(enumerate(test_loader), total=len(test_loader)):
    torch.save(images, f"precomp/val/images-batch-{i:02d}.pt")
    labels = torch.stack([t['labels'][0] for t in targets])
    torch.save(labels, f"precomp/val/labels-batch-{i:02d}.pt")
    boxes = torch.stack([t['boxes'][0] for t in targets])
    torch.save(boxes, f"precomp/val/boxes-batch-{i:02d}.pt")
