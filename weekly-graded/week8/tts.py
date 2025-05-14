from datasets import load_dataset, Audio
from dataclasses import dataclass
from functools import partial
from typing import Any, List, Union, Dict
from transformers import SpeechT5Processor, SpeechT5ForSpeechToText
from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer
from collections import Counter
import torch
from speechbrain.pretrained import EncoderClassifier
import pandas as pd
import matplotlib.pyplot as plt


def extract_chars(batch):
    all_text = " ".join(batch["normalized_text"])
    vocab = set(all_text)
    return {"vocab": [list(vocab)], "all_text": [all_text]}


def cleanup_chars(inputs, replacements):
    for src, dst in replacements:
        inputs["normalized_text"] = inputs["normalized_text"].replace(src, dst)
    return inputs


def create_speaker_embedding(wave):
    with torch.no_grad():
        emb = speaker_model.encode_batch(torch.tensor(wave))
        emb = torch.nn.functional.normalize(emb, dim=2)
    return emb.squeeze().cpu().numpy()


def prep_dataset(example, processor):
    audio = example["audio"]
    example = processor(
        text=example["normalized_text"],
        audio_target=audio["array"],
        sampling_rate=audio["sampling_rate"],
        return_attention_mask=False,
    )

    # Strip off the batch dimension
    example["labels"] = example["labels"][0]
    example["speaker_embeddings"] = create_speaker_embedding(audio["array"])
    return example


model = SpeechT5ForSpeechToText.from_pretrained("microsoft/speecht5_tts")


@dataclass
class TTSDataCollatorWithPadding:
    processor: Any

    def __call__(
        self, features: List[Dict[str, Union[List[int], torch.Tensor]]]
    ) -> Dict[str, torch.Tensor]:
        input_ids = [{"input_ids": feature["input_ids"]} for feature in features]
        label_features = [{"input_values": feature["labels"]} for feature in features]
        speaker_features = [feature["speaker_embeddings"] for feature in features]

        # collate the inputs and targets into a batch
        batch = processor.pad(
            input_ids=input_ids, labels=label_features, return_tensors="pt"
        )

        # replace padding with -100 to ignore loss correctly
        batch["labels"] = batch["labels"].masked_fill(
            batch.decoder_attention_mask.unsqueeze(-1).ne(1), -100
        )

        # not used during fine-tuning
        del batch["decoder_attention_mask"]

        # round down target lengths to multiple of reduction factor
        if model.config.reduction_factor > 1:
            target_lengths = torch.tensor(
                [len(feature["input_values"]) for feature in label_features]
            )
            target_lengths = target_lengths.new(
                [length - length % model.config.reduction_factor for length in target_lengths]
            )
            max_length = max(target_lengths)
            batch['labels'] = batch['labels'][:, :max_length]
        batch['speaker_embeddings'] = torch.tensor(speaker_features)
        return batch


ds = load_dataset("facebook/voxpopuli", "nl", split="train")

subset_size = len(ds) // 8
subset = ds.shuffle(seed=42).select(range(subset_size))

dataset = subset.cast_column("audio", Audio(sampling_rate=16_000))
processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts")
tokenizer = processor.tokenizer

vocab = dataset.map(
    extract_chars,
    batched=True,
    batch_size=-1,
    keep_in_memory=True,
    remove_columns=dataset.column_names,
)
dataset_vocab = set(vocab["vocab"][0])
tokenizer_vocab = set(tokenizer.get_vocab().keys())

print(dataset_vocab - tokenizer_vocab)  # NOQA: T201
#  {'í', 'ö', 'ï', 'ü', 'è', 'ë', 'à', ' '}
replacements = [
    ("í", "i"),
    ("ö", "o"),
    ("ï", "i"),
    ("ü", "u"),
    ("è", "e"),
    ("ë", "e"),
    ("à", "a"),
]
dataset = dataset.map(lambda x: cleanup_chars(x, replacements))

_, ax = plt.subplots()
counter = Counter(dataset["speaker_id"])
pd.Series(counter).hist(bins=20, ax=ax)
ax.set_xlabel("Examples")
ax.set_ylabel("Speakers")
plt.show()


# Select samples where speakers have between 100 and 400 samples
dataset = dataset.filter(
    lambda x: 100 <= counter[x["speaker_id"]] <= 400,
)


print(f'Total {len(set(dataset["speaker_id"]))} speakers.')  # NOQA: T201

speaker_model = EncoderClassifier.from_hparams(
    source="speechbrain/spkrec-xvect-voxceleb", savedir="/tmp/spkrec-xvect-voxceleb"
)

sample_features = prep_dataset(dataset[0], processor)
plt.imshow(sample_features["labels"].T, origin="lower")
plt.colorbar()
plt.show()


dataset = dataset.map(lambda x: prep_dataset(x, processor),
                      remove_columns=dataset.column_names)
# not too long
dataset = dataset.filter(lambda x: len(x) < 200, input_columns=["input_ids"])
dataset = dataset.train_test_split(test_size=0.1)


collator = TTSDataCollatorWithPadding(processor=processor)
model.config.use_cache = False
model.generate = partial(model.generate, use_cache=True)


training_args = Seq2SeqTrainingArguments(
    output_dir="speecht5_finetuned_voxpopuli_nl",
    per_device_train_batch_size=8,
    gradient_accumulation_steps=4,
    learning_rate=1e-4,
    warmup_steps=200,
    max_steps=600,
    gradient_checkpointing=True,
    fp16=True,
    evaluation_strategy="steps",
    per_device_eval_batch_size=4,
    save_steps=200,
    eval_steps=200,
    logging_steps=50,
    load_best_model_at_end=True,
    greater_is_better=False,
    label_names=["labels"],
    push_to_hub=False,
)

trainer = Seq2SeqTrainer(
    model=model,
    data_collator=collator,
    args=training_args,
    train_dataset=dataset['train'],
    eval_dataset=dataset['test'],
    tokenizer=processor
)
yuri = trainer.train()
