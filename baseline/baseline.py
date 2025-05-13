from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import pandas as pd

splits = {'train': 'train.csv', 'validation': 'val.csv', 'test': 'test.csv'}
df = pd.read_csv("hf://datasets/gtfintechlab/finer-ord/" + splits["train"]) #optional dataset for later, but for baseline must use provided text

def read_conllu_like_iob2(filepath):
    sentences = []
    labels = []
    sentence, label = [], []

    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line == '':
                if sentence:
                    sentences.append(sentence)
                    labels.append(label)
                    sentence, label = [], []
            elif line.startswith('#'):
                continue  # Skip metadata
            else:
                parts = line.split('\t')
                if len(parts) >= 3:  # Ensure enough columns
                    word, tag = parts[1], parts[2]
                    sentence.append(word)
                    label.append(tag)

    # Add last sentence if file doesn't end with newline
    if sentence:
        sentences.append(sentence)
        labels.append(label)

    return sentences, labels

sentences, tags = read_conllu_like_iob2('en_ewt-ud-train.iob2')
sentences2, tags2 = read_conllu_like_iob2('en_ewt-ud-dev.iob2')



def create_label_mapping(labels_list):
    unique_labels = set(label for labels in labels_list for label in labels)
    label2id = {label: i for i, label in enumerate(sorted(unique_labels))}
    id2label = {i: label for label, i in label2id.items()}
    return label2id, id2label



from torch.utils.data import Dataset
from transformers import AutoTokenizer

class NERDataset(Dataset):
    def __init__(self, sentences, labels, tokenizer, label2id, max_len=128):
        self.sentences = sentences
        self.labels = labels
        self.tokenizer = tokenizer
        self.label2id = label2id
        self.max_len = max_len

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        tokens = self.sentences[idx]
        label_tags = self.labels[idx]

        encoding = self.tokenizer(tokens,
                                 is_split_into_words=True,
                                 padding='max_length',
                                 truncation=True,
                                 max_length=self.max_len,
                                 return_tensors='pt')
        
        labels = [-100] * len(encoding['input_ids'][0])  # default ignore index for padding & subwords
        word_ids = encoding.word_ids(batch_index=0)
        previous_word_idx = None
        for i, word_idx in enumerate(word_ids):
            if word_idx is None:
                continue
            if word_idx != previous_word_idx:  # Only label first word piece
                labels[i] = self.label2id[label_tags[word_idx]]
            previous_word_idx = word_idx

        encoding['labels'] = torch.tensor(labels)
        return {key: val.squeeze(0) for key, val in encoding.items()}



from transformers import AutoModelForTokenClassification, TrainingArguments, Trainer, AutoModelForMaskedLM

# Load data
train_sentences, train_labels = read_conllu_like_iob2('en_ewt-ud-train.iob2')
dev_sentences, dev_labels = read_conllu_like_iob2('en_ewt-ud-dev.iob2')

# Label mapping
label2id, id2label = create_label_mapping(train_labels + dev_labels)
num_labels = len(label2id)

# Model
tokenizer = AutoTokenizer.from_pretrained("FinanceInc/finbert-pretrain")
model = AutoModelForTokenClassification.from_pretrained(
    "FinanceInc/finbert-pretrain",
    num_labels=len(label2id),
    id2label=id2label,
    label2id=label2id
) # this works, some other finBERT have questionable success, but can research otherbaseline options

train_dataset = NERDataset(train_sentences, train_labels, tokenizer, label2id)
dev_dataset = NERDataset(dev_sentences, dev_labels, tokenizer, label2id)


training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=0.0001,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
    save_strategy="epoch",
    logging_dir="./logs",
    logging_steps=10,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=dev_dataset,
    tokenizer=tokenizer
)

trainer.train()

# Save model weights and tokenizer to a directory
save_path = "./baseline_model"  
trainer.save_model(save_path)
tokenizer.save_pretrained(save_path)
