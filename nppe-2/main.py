from tqdm import tqdm
from datasets import load_dataset
from torch.utils.data import DataLoader
import torch
from torch import nn
from transformers import Wav2Vec2Model, Wav2Vec2Processor
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

torch.manual_seed(42)
device = torch.device("cuda")

ds = load_dataset("facebook/voxpopuli", "sl", split="train", streaming=True).shuffle(seed=42)
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
feat_extract = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h").to(device)


def feat(sample, padding=True):
    enc = processor(
        sample["audio"]["array"],
        sampling_rate=16_000,
        return_tensors="pt",
        padding=padding,
    )
    enc['input_values'] = enc['input_values'].to(device)
    with torch.no_grad():
        output = feat_extract(**enc)
    return {
        "features": output.last_hidden_state[0],
        "n_frames": output.last_hidden_state.shape[1],
        "labels": sample["speaker_id"],
    }


ds_filtered = ds.map(
    feat, remove_columns=ds.column_names
).filter(lambda x: x["n_frames"] >= 200)


class SpeakerClassification(nn.Module):

    def __init__(self, n_classes, *args, **kwargs):
        super(SpeakerClassification, self).__init__(*args, **kwargs)

        self.relu = nn.ReLU()

        # Conv1: In channels: 768, Out channels: 256, Kernel size: 3
        # BatchNorm1: Features: 256
        # ReLU1: Applied after BatchNorm1
        # MaxPool1: Kernel size: 2
        self.conv1 = nn.Conv1d(in_channels=768, out_channels=256, kernel_size=3)
        self.bnorm1 = nn.BatchNorm1d(256)
        self.maxpool1 = nn.MaxPool1d(kernel_size=2)

        # Conv2: In channels: 256, Out channels: 128, Kernel size: 3
        # BatchNorm2: Features: 128
        # ReLU2: Applied after BatchNorm2
        # MaxPool2: Kernel size: 2
        self.conv2 = nn.Conv1d(in_channels=256, out_channels=128, kernel_size=3)
        self.bnorm2 = nn.BatchNorm1d(128)
        self.maxpool2 = nn.MaxPool1d(kernel_size=2)

        # Conv3: In channels: 128, Out channels: 32, Kernel size: 3
        # BatchNorm3: Features: 32
        # ReLU3: Applied after BatchNorm3
        # MaxPool3: Kernel size: 2
        self.conv3 = nn.Conv1d(in_channels=128, out_channels=32, kernel_size=3)
        self.bnorm3 = nn.BatchNorm1d(32)
        self.maxpool3 = nn.MaxPool1d(kernel_size=2)

        # Global Average Pooling: Averaged across the temporal dimension
        # FC1: Input features: 32, Output features: 128
        # ReLU4: Applied after FC1
        # FC2: Input features: 128, Output features: Number of classes
        self.fc1 = nn.Linear(in_features=32, out_features=128)
        self.fc2 = nn.Linear(in_features=128, out_features=n_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bnorm1(x)
        x = self.relu(x)
        x = self.maxpool1(x)

        x = self.conv2(x)
        x = self.bnorm2(x)
        x = self.relu(x)
        x = self.maxpool2(x)

        x = self.conv3(x)
        x = self.bnorm3(x)
        x = self.relu(x)
        x = self.maxpool3(x)

        x = torch.mean(x, axis=-1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


print('Defined model...')

xy = [(sample['features'], sample['labels']) for sample in ds_filtered]
xtrain, xtest = train_test_split(xy, test_size=0.2, random_state=42, stratify=[l[1] for l in xy])
xtrain, xval = train_test_split(xtrain, test_size=0.1, random_state=42,
                                stratify=[l[1] for l in xtrain])

speakers = list(set([sample[1] for sample in xy]))
speakers = sorted(speakers)
spk_mapping = {spk: i for i, spk in enumerate(speakers)}

print("Datasets ready...")


def collate(batch):
    feats = torch.stack([sample[0][:200] for sample in batch])
    labels = torch.tensor([spk_mapping[sample[1]] for sample in batch])
    return feats.transpose(2, 1), labels


train_dataloader = DataLoader(xtrain, batch_size=100, collate_fn=collate)
test_dataloader = DataLoader(xtest, batch_size=1, collate_fn=collate)
val_dataloader = DataLoader(xval, batch_size=10, collate_fn=collate)


model = SpeakerClassification(n_classes=len(spk_mapping))
model.to(device)

losser = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


def train(model, losser, optimizer, test_dataloader, val_dataloader, n_epochs=10):
    for i in tqdm(range(n_epochs), desc="Epoch"):
        model.train()
        epoch_train_loss = 0

        y_true_train = []
        y_pred_train = []

        # for batch in tqdm(train_dataloader, total=len(train_dataloader), desc='Train batch', leave=False):
        for batch, labels in train_dataloader:
            optimizer.zero_grad()
            outputs = model(batch.to(device))
            loss = losser(outputs, labels.to(device))
            loss.backward()
            optimizer.step()
            batch_loss = loss.item()
            epoch_train_loss += batch_loss

            y_true = labels.tolist()
            y_true_train.extend(y_true)
            y_pred = torch.argmax(nn.functional.softmax(outputs, dim=1), axis=1).tolist()
            y_pred_train.extend(y_pred)
            # print(f'\tBatch train loss: {batch_loss}; Batch train accuracy: {round(accuracy_score(y_true, y_pred), 2)}')
        epoch_train_loss = epoch_train_loss / len(train_dataloader)

        # Validation
        y_true_val = []
        y_pred_val = []
        model.eval()
        epoch_val_loss = 0
        with torch.no_grad():
            # for batch in tqdm(val_dataloader, total=len(val_dataloader), desc='Val batch', leave=False):
            for batch, labels in val_dataloader:
                output = model(batch.to(device))
                loss = losser(output, labels.to(device))
                batch_val_loss = loss.item()
                epoch_val_loss += batch_val_loss
                y_true = labels.tolist()
                y_true_val.extend(y_true)
                y_pred = torch.argmax(output, axis=1).tolist()
                y_pred_val.extend(y_pred)
                # print(f'\tBatch val loss: {batch_val_loss}; Batch val accuracy: {round(accuracy_score(y_true, y_pred), 2)}')

        train_acc = round(accuracy_score(y_true_train, y_pred_train), 2)
        val_acc = round(accuracy_score(y_true_val, y_pred_val), 2)
        epoch_val_loss = epoch_val_loss / len(val_dataloader)
        print(
            f"Epoch {i}: Train loss: {round(epoch_train_loss, 3)}; Train acc: {train_acc}; Val loss: {round(epoch_val_loss, 3)}; Val acc: {val_acc}"
        )


train(model, losser, optimizer, test_dataloader, val_dataloader, n_epochs=100)

# Find test accuracy
y_pred = []
y_true = []
with torch.no_grad():
    model.eval()
    for batch, labels in test_dataloader:
        output = model(batch.to(device))
        y_true.extend(labels.tolist())
        y_pred.extend(torch.argmax(output, axis=1).tolist())
test_acc = accuracy_score(y_true, y_pred)
print('Test Accuracy:', test_acc)
