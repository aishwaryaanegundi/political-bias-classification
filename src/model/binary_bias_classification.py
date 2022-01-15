import torch
from datasets import load_dataset
from datasets import load_metric
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from datasets import Dataset

from transformers import AutoModelForSequenceClassification
from transformers import AdamW
from transformers import get_scheduler
from tqdm.auto import tqdm
import pandas as pd

torch.cuda.empty_cache()

def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)

num_epochs = 5
batch_size = 2
learning_rate = 5e-5
path_to_train_data = './data/polly/polly_train.csv'
path_to_test_data = './data/polly/polly_test.csv'
path_to_save_model = './models/bertbasegermancased-two_extremesv3.pt'
pretrained_model_name = "deepset/gelectra-large"

raw_datasets = load_dataset('csv', data_files = path_to_train_data, cache_dir = None, split='train')
print(raw_datasets)
tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name, normalization=True)

tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)
tokenized_datasets = tokenized_datasets.remove_columns(["Tweet"])
tokenized_datasets = tokenized_datasets.remove_columns(["text"])
tokenized_datasets = tokenized_datasets.remove_columns(["Likes"])
tokenized_datasets = tokenized_datasets.remove_columns(["Type"])
tokenized_datasets = tokenized_datasets.remove_columns(["Party"])
tokenized_datasets = tokenized_datasets.remove_columns(["Unnamed: 0"])
tokenized_datasets.set_format("torch")
small_train_dataset = tokenized_datasets.shuffle(seed=42).select(range(14732)) #72121
print(small_train_dataset)

def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)

test_data = pd.read_csv(path_to_test_data)
print(test_data.columns)
test_data = test_data[['Tweet','labels']]
test_data = test_data.rename(columns={'Tweet':'text'})
test_set = Dataset.from_pandas(test_data)
print(test_set)
test_tokenised = test_set.map(tokenize_function, batched=True)
test_tokenised.set_format("torch")
print(test_tokenised)
small_test_dataset = test_tokenised.select(range(1638))
small_test_dataset.to_csv('./test.csv')
print(small_test_dataset)

train_dataloader = DataLoader(small_train_dataset, shuffle=True, batch_size=batch_size)
test_dataloader = DataLoader(small_test_dataset, batch_size=batch_size)

model = AutoModelForSequenceClassification.from_pretrained(pretrained_model_name, num_labels=2)
optimizer = AdamW(model.parameters(), lr=learning_rate)
num_training_steps = num_epochs * len(train_dataloader)
lr_scheduler = get_scheduler(
    "linear",
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=num_training_steps
)

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print('device: ', device)
model.to(device)


progress_bar = tqdm(range(num_training_steps))

model.train()
for epoch in range(num_epochs):
    for batch in train_dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()

        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
        progress_bar.update(1)

torch.save(model.state_dict(), path_to_save_model)

metric= load_metric("accuracy")
model.eval()
for batch in test_dataloader:
    batch = {k: v.to(device) for k, v in batch.items()}
    with torch.no_grad():
        outputs = model(**batch)

    logits = outputs.logits
    predictions = torch.argmax(logits, dim=-1)
    metric.add_batch(predictions=predictions, references=batch["labels"])

score = metric.compute()
print(score)