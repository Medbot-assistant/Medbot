import torch
import transformers
from transformers import BloomTokenizerFast, BloomForCausalLM, TrainingArguments

from datasets import load_dataset

from utils import ModifiedTrainer, tokenise_data, data_collator

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model_name = "bloom-1b7"
model = BloomForCausalLM.from_pretrained(f"bigscience/{model_name}")
tokeniser = BloomTokenizerFast.from_pretrained(f"bigscience/{model_name}", add_prefix_space=True)

dataset = load_dataset('dataset.tsv')
input_ids = tokenise_data(dataset, tokeniser)

model.gradient_checkpointing_enable()
model.is_parallelizable = True
model.model_parallel = True

training_args = TrainingArguments(
    "output",
    fp16=False,
    gradient_accumulation_steps= 1,
    per_device_train_batch_size = 2,
    learning_rate = 2e-5,
    num_train_epochs=2,
    logging_steps=10,
)

trainer = ModifiedTrainer(
    model=model,
    train_dataset=input_ids,
    args=training_args,
    data_collator=data_collator,
)

trainer.train()