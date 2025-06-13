# from transformers import AutoTokenizer
# from transformers import AutoModel


# checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"
# tokenizer = AutoTokenizer.from_pretrained(checkpoint)
#
# raw_inputs = [
#     "I've been waiting for a HuggingFace course my whole life.",
#     "I hate this so much!",
# ]
# inputs = tokenizer(raw_inputs, padding=True, truncation=True, return_tensors="pt")
# print(inputs)
#
#
#
# checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"
# model = AutoModel.from_pretrained(checkpoint)
#
# outputs = model(**inputs)
#
# print(outputs.last_hidden_state.shape)


# import torch
# from torch import nn
# from transformers import BertModel,BertConfig,BertTokenizer
#
# # 预训练模型存储位置
# pretrained_path = 'bert-base-chinese'
# config = BertConfig.from_pretrained(pretrained_path)
# tokenizer = BertTokenizer.from_pretrained(pretrained_path)
# model = BertModel.from_pretrained(pretrained_path,config=config)
#
# batch = tokenizer.encode_plus('这是一个晴朗的早晨') # encode_plus 返回一个字典
# input_ids = torch.tensor([batch['input_ids']])
# token_type_ids = torch.tensor([batch['token_type_ids']])
# attention_mask = torch.tensor([batch['attention_mask']])
#
# embedding = model(input_ids,token_type_ids=token_type_ids)
#
# print(embedding)


# import torch
# from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
#
# tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
# model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased")
#
# inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")
# with torch.no_grad():
#     logits = model(**inputs).logits
#
# predicted_class_id = logits.argmax().item()
# var = model.config.id2label[predicted_class_id]
#
# print(var)


# from transformers import AutoModelForSequenceClassification
# from transformers import AutoTokenizer
#
# checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"
# tokenizer = AutoTokenizer.from_pretrained(checkpoint)
#
# checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"
# model = AutoModelForSequenceClassification.from_pretrained(checkpoint)
#
# raw_inputs = [
#     "I've been waiting for a HuggingFace course my whole life.",
#     "I hate this so much!",
# ]
# inputs = tokenizer(raw_inputs, padding=True, truncation=True, return_tensors="pt")
# print(inputs)
#
# outputs = model(**inputs)
# print(outputs)


# from datasets import load_dataset
# from transformers import AutoTokenizer
# from transformers import DataCollatorWithPadding
#
# checkpoint = "bert-base-uncased"
# tokenizer = AutoTokenizer.from_pretrained(checkpoint)
#
# raw_datasets = load_dataset("glue", "mrpc")
# # print(raw_datasets)
#
# raw_train_dataset = raw_datasets["train"]
# print(raw_train_dataset[0])


# tokenized_sentences_1 = tokenizer(raw_datasets["train"]["sentence1"])
# tokenized_sentences_2 = tokenizer(raw_datasets["train"]["sentence2"])

# print(tokenized_sentences_1)

# tokenized_dataset = tokenizer(
#     raw_datasets["train"]["sentence1"],
#     raw_datasets["train"]["sentence2"],
#     padding=True,
#     truncation=True,
# )


# def tokenize_function(example):
#     return tokenizer(example["sentence1"], example["sentence2"], truncation=True)
#
# tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)
# print(tokenized_datasets)
#
# data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
#
# samples = tokenized_datasets["train"][:8]
# samples = {k: v for k, v in samples.items() if k not in ["idx", "sentence1", "sentence2"]}
# print([len(x) for x in samples["input_ids"]])
#
# batch = data_collator(samples)
#
# print({k: v.shape for k, v in batch.items()})

#
#
# # Same as before
# checkpoint = "bert-base-uncased"
# tokenizer = AutoTokenizer.from_pretrained(checkpoint)
# model = AutoModelForSequenceClassification.from_pretrained(checkpoint)
# sequences = [
#     "I've been waiting for a HuggingFace course my whole life.",
#     "This course is amazing!",
# ]
# batch = tokenizer(sequences, padding=True, truncation=True, return_tensors="pt")
#
# # This is new
# batch["labels"] = torch.tensor([1, 1])
#
# optimizer = AdamW(model.parameters())
# loss = model(**batch).loss
# loss.backward()
# optimizer.step()

from datasets import load_dataset
from transformers import AutoTokenizer, DataCollatorWithPadding
from transformers import TrainingArguments
from transformers import AutoModelForSequenceClassification
from transformers import Trainer


raw_datasets = load_dataset("glue", "mrpc")
checkpoint = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)


def tokenize_function(example):
    return tokenizer(example["sentence1"], example["sentence2"], truncation=True)


training_args = TrainingArguments("test-trainer")
model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=2)

tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

trainer = Trainer(
    model,
    training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    data_collator=data_collator,
    tokenizer=tokenizer,
)


trainer.train()
