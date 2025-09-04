from transformers import AutoModelForCausalLM, TrainingArguments, Trainer, AutoTokenizer
import torch
import datasets
import numpy as np

import peft

MODEL_NAME = "google/gemma-2b"
SEQ_LEN = 350
OUT_DIR = 'data'

def train_lora():
    # model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype='auto')
    print(model)
    model_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total trainable parameters in model : {model_params}")

    lora_model = peft.get_lora_model(model)
    print(lora_model)
    lora_model_params = sum(p.numel() for p in lora_model.parameters() if p.requires_grad)
    print(f"Total trainable parameters in lora model : {lora_model_params} and are {(lora_model_params/model_params)*100} % of the original model")
    
    # dataset = getDataset()
    dataset = get_dataset_gsm8k()
    train(lora_model, dataset, OUT_DIR)
    
class ModelWithLoss(torch.nn.Module): 
    def __init__(self, model):
        super().__init__()
        self.model = model
        # self.loss = torch.nn.CrossEntropyLoss(reduction="none")
        # self.loss = torch.nn.CrossEntropyLoss(reduction="sum")
        self.loss = torch.nn.CrossEntropyLoss()
    def forward(self, *, input_ids=None, attention_mask=None ):
        result = self.model(input_ids=input_ids, attention_mask=attention_mask)
        logits = result['logits'][...,:-1,:]
        result.loss = self.loss( logits.reshape(-1, logits.shape[-1]), input_ids[:,1:].reshape(-1) )
        return (result.loss,)
        # return result
    
def train(model, lm_dataset, output_dir):
    # model = ModelWithLoss(model)
    
    training_args = TrainingArguments(
        output_dir=output_dir,
        eval_strategy="epoch",
        learning_rate=2e-5,
        weight_decay=0.01,
        # torch_compile=True,
        save_strategy='no',
        per_device_train_batch_size = 1,
        per_device_eval_batch_size = 1,
        save_safetensors = False, # work around bug 
        # bf16=True,
        gradient_accumulation_steps=8,
        logging_steps=50
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=lm_dataset["train"],
        eval_dataset=lm_dataset["test"],
        # data_collator=data_collator,
    )

    trainer.train()
    return trainer


def get_dataset_gsm8k():
    from datasets import load_dataset
    data = load_dataset('openai/gsm8k', 'main')
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    #split data
    # data = data["train"].train_test_split(test_size=.2, seed=1)
    
    def preprocess(batch):
        def format_question(ex):
            return f"Q: {ex}\nA: "

        def format_answer(ex):
            # this is what helm does
            answer_text = ex.replace("####", "The answer is").replace("\n", " ") + "."
            return f"{answer_text}\n{tokenizer.eos_token}"

        sources = [format_question(question) for question in batch['question']]
        targets = [format_answer(answer) for answer in batch['answer']]

        examples = [s + t for s, t in zip(sources, targets)]
        sources_tokenized = tokenizer(sources, return_tensors="np", padding=False, truncation=True, max_length=SEQ_LEN)
        examples_tokenized = tokenizer(examples, return_tensors="np", padding=False, truncation=True, max_length=SEQ_LEN)
        # examples_tokenized = tokenizer(examples, return_tensors="np", padding='max_length', truncation=True, max_length=SEQ_LEN)

        source_lens = [len(s) for s in sources_tokenized["input_ids"]]

        return {
            "input_ids": examples_tokenized["input_ids"],
            "labels": examples_tokenized["input_ids"],
            # "attention_mask": torch.ones(examples_tokenized["input_ids"].shape),
            "source_lens": source_lens,
        }
    
    return data.map(preprocess, batched=True)

if __name__ == '__main__':
    trainer = train_lora()
    

