# coding: utf-8
from datasets import load_dataset
from transformers import (
    AutoModelForCTC, Wav2Vec2CTCTokenizer, Wav2Vec2FeatureExtractor,
    Wav2Vec2Processor
)
from transformers import Trainer, TrainingArguments
import json
from dataclasses import dataclass
from evaluate import load
import numpy as np

ds = load_dataset("springlab/asr-task-data")


def extract_all_chars(batch):
    all_text = " ".join(batch["text"])
    vocab = list(set(all_text))
    return {"vocab": [vocab], "all_text": [all_text]}


def prep_dataset(batch):
    audio = batch["audio"]
    batch["input_values"] = processor(
        audio["array"], sampling_rate=audio["sampling_rate"]
    ).input_values[0]
    batch["input_length"] = len(batch["input_values"])
    with processor.as_target_processor():
        batch["labels"] = processor(batch["text"]).input_ids
    return batch


def compute_metrics(pred):
    pred_ids = np.argmax(pred.predictions, axis=-1)
    pred.label_ids[pred.label_ids == -100] = processor.tokenizer.pad_token_id
    pred_str = processor.batch_decode(pred_ids)
    label_str = processor.batch_decode(pred.label_ids, group_tokens=False)
    return {"wer": wer.compute(predictions=pred_str, references=label_str)}


@dataclass
class DataCollatorCTCWithPadding:
    processor: Wav2Vec2Processor
    padding: bool = True

    def __call__(self, features):
        input_features = [
            {"input_values": feature["input_values"]} for feature in features
        ]
        label_features = [
            {"input_ids": feature["labels"]} for feature in features
        ]
        batch = self.processor.pad(
            input_features, padding=self.padding, return_tensors="pt"
        )
        with self.processor.as_target_processor():
            labels_batch = self.processor.pad(
                label_features, padding=self.padding, return_tensors='pt'
            )
        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100
        )
        batch["labels"] = labels
        return batch


vocab = ds.map(extract_all_chars, batch_size=8)
vlist = []
for v in vocab["train"]["vocab"]:
    vlist.extend(v[0])


vlist = set(vlist)
vdict = {v: k for k, v in enumerate(vlist)}
vdict["|"] = vdict.pop(" ")
vdict["[UNK]"] = vdict["[PAD]"] = len(vdict)
with open("vocab.json", "w") as fout:
    json.dump(vdict, fout, indent=2)


tokenizer = Wav2Vec2CTCTokenizer(
    "./vocab.json", unk_token="[UNK]", pad_token="[PAD]",
    word_delimiter_token="|"
)

feat_extractor = Wav2Vec2FeatureExtractor(
    feature_size=1,
    sampling_rate=16_000,
    padding_value=0.0,
    do_normalize=True,
    return_attention_mask=True,
)
processor = Wav2Vec2Processor(feature_extractor=feat_extractor,
                              tokenizer=tokenizer)
ds_proc = ds.map(prep_dataset, batch_size=16, num_proc=9)
collator = DataCollatorCTCWithPadding(processor=processor, padding=True)

wer = load("wer")
dataset = ds_proc["train"].train_test_split(
    test_size=0.1, shuffle=True, seed=42
)

model = AutoModelForCTC.from_pretrained(
    "facebook/wav2vec2-base",
    ctc_loss_reduction="mean",
    pad_token_id=processor.tokenizer.pad_token_id,
    vocab_size=len(processor.tokenizer),
)

model.freeze_feature_encoder()

tr_args = TrainingArguments(
    output_dir="ASR_task",
    group_by_length=True,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    eval_strategy="steps",
    num_train_epochs=7,
    fp16=True,
    gradient_checkpointing=True,
    save_steps=1000,
    eval_steps=500,
    learning_rate=1e-4,
    weight_decay=0.005,
    warmup_steps=1000,
    save_total_limit=2,
    load_best_model_at_end=True,
    save_strategy="steps",
)
trainer = Trainer(
    model=model,
    data_collator=collator,
    args=tr_args,
    compute_metrics=compute_metrics,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    tokenizer=processor.feature_extractor,
)
yuri = trainer.train()
