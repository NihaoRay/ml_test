import torch
from datasets import load_dataset
from transformers import AutoTokenizer, DataCollatorWithPadding
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification
from transformers import AdamW
from transformers import get_scheduler
from tqdm.auto import tqdm
import evaluate
from sklearn.metrics import f1_score, roc_auc_score, recall_score, precision_score

# 读取数据集
raw_datasets = load_dataset("glue", "mrpc")
checkpoint = "bert-base-uncased"

# 数据分词器
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

def tokenize_function(example):
    return tokenizer(example["sentence1"], example["sentence2"], truncation=True)

tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# 删除某些无用的特征
tokenized_datasets = tokenized_datasets.remove_columns(["sentence1", "sentence2", "idx"])
tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
tokenized_datasets.set_format("torch")

print(tokenized_datasets["train"].column_names)

# 加入读取
train_dataloader = DataLoader(
    tokenized_datasets["train"], shuffle=True, batch_size=8, collate_fn=data_collator
)
eval_dataloader = DataLoader(
    tokenized_datasets["validation"], batch_size=8, collate_fn=data_collator
)

# 架子模型
model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=2)

# 添加优化器
optimizer = AdamW(model.parameters(), lr=5e-5)

num_epochs = 3
num_training_steps = num_epochs * len(train_dataloader)
lr_scheduler = get_scheduler(
    "linear",
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=num_training_steps,
)

print(num_training_steps)

# 测试的代码
def evaluate_f1():
    # metric = evaluate.load("glue", "mrpc")
    # print(metric)
    model.eval()
    score_preds = []
    score_label = []
    for batch in eval_dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            outputs = model(**batch)

        logits = outputs.logits
        predictions = torch.argmax(logits, dim=-1)
        score_preds.extend(predictions.cpu().numpy())
        score_label.extend(batch["labels"].cpu().numpy())

        # metric.add_batch(predictions=predictions, references=batch["labels"])

    print(f"f1:{f1_score(score_label, score_preds)}, recall:{recall_score(score_label, score_preds)}, precision_score:{precision_score(score_label, score_preds)}")
    # print(metric.compute())

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model.to(device)

# 训练循环，测试的代码
progress_bar = tqdm(range(num_training_steps))
model.train()

for epoch in range(num_epochs):
    print(f"epcho:{epoch+1}")
    for batch in train_dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()

        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
        progress_bar.update(1)
    # 添加测试代码
    evaluate_f1()


