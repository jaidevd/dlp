import torch
from torch.utils.data import Dataset, DataLoader
import os
import fcos
from collections import defaultdict
from tqdm import tqdm

op = os.path
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class TrainDataset(Dataset):

    def __init__(self, root, max_batch_size, n_samples):
        self.root = root
        files = os.listdir(root)
        self.major_batches = map(int, set([
            op.splitext(k.split('-')[-1])[0] for k in files
        ]))
        self.current_batch_ix = None
        self.image_batch = None
        self.boxes_batch = None
        self.labels_batch = None
        self.max_batch_size = max_batch_size
        self.n_samples = n_samples

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        reqd_batch = idx // self.max_batch_size
        if reqd_batch != self.current_batch_ix:
            self.current_batch_ix = reqd_batch
            path = f'{self.root}/images-batch-{reqd_batch:02d}.pt'
            self.image_batch = torch.load(path, weights_only=True)
            path = f'{self.root}/boxes-batch-{reqd_batch:02d}.pt'
            self.boxes_batch = torch.load(path, weights_only=True)
            path = f'{self.root}/labels-batch-{reqd_batch:02d}.pt'
            self.labels_batch = torch.load(path, weights_only=True)
        newix = idx % self.max_batch_size
        return self.image_batch[newix], self.boxes_batch[newix], self.labels_batch[newix].item()


def collate(batch):
    images, boxes, labels = zip(*batch)
    targets = []
    for b, l in zip(boxes, labels):
        targets.append({
            'boxes': b.unsqueeze(0).to(device),
            'labels': torch.tensor([l], dtype=torch.int64).to(device)
        })
    return torch.stack(images).to(device), targets


def train(model, train_loader, test_loader, n_epochs=1, patience=3):
    opt = torch.optim.Adam(model.head.classification_head.parameters(), lr=0.001)
    lr = torch.optim.lr_scheduler.StepLR(opt, step_size=10, gamma=0.05)
    b_len = len(train_loader) + len(test_loader)
    epoch_history = defaultdict(list)  # NOQA: F841
    with tqdm(desc="Epoch", total=n_epochs, bar_format=fcos.BAR_FORMAT) as ebar:
        for epoch in range(n_epochs):
            model.train()
            epoch_train_loss = epoch_val_loss = 0
            epoch_val_metric = 0
            with tqdm(
                desc="Batch", total=b_len, bar_format=fcos.BAR_FORMAT, leave=False
            ) as bbar:
                for images, targets in train_loader:
                    opt.zero_grad()
                    loss_dict = model(images, targets)
                    batch_train_loss = sum(loss for loss in loss_dict.values())
                    batch_train_loss.backward()
                    btl = batch_train_loss.item()
                    opt.step()
                    bbar.update(1)
                    bbar.set_postfix_str(
                        f"Train(loss={btl:.3f})", refresh=True
                    )
                    epoch_train_loss += btl
                epoch_train_loss /= len(train_loader)
                for images, targets in test_loader:
                    with torch.no_grad():
                        preds = model(images, targets)
                    batch_val_loss = sum(loss for loss in preds.values())
                    bbar.set_postfix_str(
                        f"Train(loss={epoch_train_loss:.3f});Val(loss={batch_val_loss:.3f})",
                        refresh=True
                    )
                    epoch_val_loss += batch_val_loss.item()
                    bbar.update(1)
                epoch_val_loss /= len(test_loader)
                epoch_val_metric /= len(test_loader)
            lr.step()
            report = fcos.update_history(
                epoch_history,
                train_loss=epoch_train_loss,
                val_loss=epoch_val_loss,
                val_metric=epoch_val_metric,
                patience=patience,
            )
            fcos.update_postfix(ebar, report, lr.get_last_lr())
    return epoch_history


if __name__ == "__main__":
    import json
    model = fcos.get_model()
    dstrain, dstest = TrainDataset('precomp/train', 256, 6000), TrainDataset('precomp/val', 128, 1500)
    train_loader = DataLoader(dstrain, shuffle=False, batch_size=16, collate_fn=collate)
    test_loader = DataLoader(dstest, shuffle=False, batch_size=16, collate_fn=collate)
    history = train(model, train_loader, test_loader, n_epochs=50, patience=5)
    with open("history.json", "w") as f_out:
        json.dump(history, f_out, indent=2)
    torch.save(model.state_dict(), "fcos-50-epochs-lr.pth")
    # model = load_model("fcos-10-epochs.pth")
    fcos.make_submission(model, "data/test/images/", show=4)
