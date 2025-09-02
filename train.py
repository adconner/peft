from transformers import AutoModelForCausalLM, TrainingArguments, Trainer, AutoTokenizer
import torch
import datasets
import numpy as np

from functools import partial
import time

MODEL_NAME = "google/gemma-2b"
SEQ_LEN = 540
OUT_DIR = 'data'
    
class LinearWithLoRA(torch.nn.Module):
    def __init__(self, linear, rank, alpha):
        super().__init__()
        self.linear = linear
        std_dev = 1 / np.sqrt(rank)
        self.A = torch.nn.Parameter(torch.randn(linear.in_features, rank, dtype = torch.bfloat16) * std_dev)
        self.B = torch.nn.Parameter(torch.zeros(rank, linear.out_features, dtype = torch.bfloat16 ))

    def forward(self, x):
        return self.linear(x) + 16 * x @ self.A @ self.B

def get_lora_model(model):
    assign_lora = partial(LinearWithLoRA, rank=8, alpha=16)

    for param in model.parameters():
        param.requires_grad = False

    for layer in model.model.layers:
        layer.self_attn.q_proj = assign_lora(layer.self_attn.q_proj)
        layer.self_attn.k_proj = assign_lora(layer.self_attn.k_proj)
        layer.self_attn.v_proj = assign_lora(layer.self_attn.v_proj)
        layer.self_attn.o_proj = assign_lora(layer.self_attn.o_proj)
        layer.mlp.gate_proj = assign_lora(layer.mlp.gate_proj)
        layer.mlp.up_proj = assign_lora(layer.mlp.up_proj)
        layer.mlp.down_proj = assign_lora(layer.mlp.down_proj)
    # model.lm_head = assign_lora(model.lm_head)

    return model

def train_lora():
    # model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype='auto')
    print(model)
    model_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total trainable parameters in model : {model_params}")

    lora_model = get_lora_model(model)
    print(lora_model)
    lora_model_params = sum(p.numel() for p in lora_model.parameters() if p.requires_grad)
    print(f"Total trainable parameters in lora model : {lora_model_params} and are {(lora_model_params/model_params)*100} % of the original model")
    
    # dataset = getDataset()
    dataset = get_dataset_gsm8k()
    train(lora_model, dataset, OUT_DIR)
    
def train(model, lm_dataset, output_dir):
    
    training_args = TrainingArguments(
        output_dir=output_dir,
        eval_strategy="epoch",
        learning_rate=2e-5,
        weight_decay=0.01,
        # torch_compile=True,
        save_steps=100000,
        per_device_train_batch_size = 1,
        per_device_eval_batch_size = 1,
        save_safetensors = False, # work around bug 
        # bf16=True
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=lm_dataset["train"],
        eval_dataset=lm_dataset["test"],
        # data_collator=data_collator,
    )

    st = time.time()
    trainer.train()
    et = time.time()
    print(f"total training time : {(et - st)} sec.")
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
