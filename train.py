from transformers import AutoModelForCausalLM, TrainingArguments, Trainer, AutoTokenizer
import torch
import datasets
import numpy as np

from functools import partial
import time

MODEL_NAME = "google/gemma-2b"
SEQ_LEN = 350
OUT_DIR = 'data'
    
def get_lora_model(model):
    class LinearWithLoRA(torch.nn.Module):
        def __init__(self, linear, rank):
            super().__init__()
            assert linear.bias is None
            self.linear = linear
            self.A = torch.nn.Parameter(torch.randn(linear.in_features, rank, dtype = torch.bfloat16)/np.sqrt(rank))
            self.B = torch.nn.Parameter(torch.zeros(rank, linear.out_features, dtype = torch.bfloat16 ))
        def forward(self, x):
            return self.linear(x) + 16 * x @ self.A @ self.B
    wrap_linear = partial(LinearWithLoRA, rank=8)
    for param in model.parameters():
        param.requires_grad = False
    for layer in model.model.layers:
        layer.self_attn.q_proj = wrap_linear(layer.self_attn.q_proj)
        layer.self_attn.k_proj = wrap_linear(layer.self_attn.k_proj)
        layer.self_attn.v_proj = wrap_linear(layer.self_attn.v_proj)
        layer.self_attn.o_proj = wrap_linear(layer.self_attn.o_proj)
        layer.mlp.gate_proj = wrap_linear(layer.mlp.gate_proj)
        layer.mlp.up_proj = wrap_linear(layer.mlp.up_proj)
        layer.mlp.down_proj = wrap_linear(layer.mlp.down_proj)
    # need to do this simultaneously with lm_head and embedding
    # model.lm_head = wrap_linear(model.lm_head)
    return model

def get_dora_model(model):
    class LinearWithDora(torch.nn.Module):
        def __init__(self, linear, rank):
            super().__init__()
            assert linear.bias is None
            W = linear.weight.T
            mag = torch.linalg.norm(W, dim=0, keepdim=True)
            self.W = torch.nn.Parameter(W / mag, requires_grad = False)
            self.mag = torch.nn.Parameter(mag)
            self.A = torch.nn.Parameter(torch.randn(linear.in_features, rank, dtype = torch.bfloat16)/np.sqrt(rank))
            self.B = torch.nn.Parameter(torch.zeros(rank, linear.out_features, dtype = torch.bfloat16 ))
        def forward(self, x):
            norm = torch.linalg.norm(self.W + 16 * self.A @ self.B, dim=0, keepdim=True)
            return (x @ self.W + 16 * x @ self.A @ self.B) * self.mag / norm
            # W = self.W + 16 * self.A @ self.B
            # W *= self.mag / torch.linalg.norm(W, dim=0, keepdim=True)
            # return x @ W
    wrap_linear = partial(LinearWithDora, rank=8)
    for param in model.parameters():
        param.requires_grad = False
    for layer in model.model.layers:
        layer.self_attn.q_proj = wrap_linear(layer.self_attn.q_proj)
        layer.self_attn.k_proj = wrap_linear(layer.self_attn.k_proj)
        layer.self_attn.v_proj = wrap_linear(layer.self_attn.v_proj)
        layer.self_attn.o_proj = wrap_linear(layer.self_attn.o_proj)
        layer.mlp.gate_proj = wrap_linear(layer.mlp.gate_proj)
        layer.mlp.up_proj = wrap_linear(layer.mlp.up_proj)
        layer.mlp.down_proj = wrap_linear(layer.mlp.down_proj)
    # need to do this simultaneously with lm_head and embedding
    # model.lm_head = wrap_linear(model.lm_head)
    return model

def get_simple_dora_model(model):
    class LinearWithSimpleDoraTranspose(torch.nn.Module):
        def __init__(self, linear, rank):
            super().__init__()
            assert linear.bias is None
            W = linear.weight.T
            mag = torch.linalg.norm(W, dim=0, keepdim=True)
            self.W = torch.nn.Parameter(W / mag, requires_grad = False)
            self.mag = torch.nn.Parameter(mag)
            self.A = torch.nn.Parameter(torch.randn(linear.in_features, rank, dtype = torch.bfloat16)/np.sqrt(rank))
            self.B = torch.nn.Parameter(torch.zeros(rank, linear.out_features, dtype = torch.bfloat16 ))
        def forward(self, x):
            return (x @ self.W + 16 * x @ self.A @ self.B) * self.mag
    wrap_linear = partial(LinearWithSimpleDoraTranspose, rank=8)
    for param in model.parameters():
        param.requires_grad = False
    for layer in model.model.layers:
        layer.self_attn.q_proj = wrap_linear(layer.self_attn.q_proj)
        layer.self_attn.k_proj = wrap_linear(layer.self_attn.k_proj)
        layer.self_attn.v_proj = wrap_linear(layer.self_attn.v_proj)
        layer.self_attn.o_proj = wrap_linear(layer.self_attn.o_proj)
        layer.mlp.gate_proj = wrap_linear(layer.mlp.gate_proj)
        layer.mlp.up_proj = wrap_linear(layer.mlp.up_proj)
        layer.mlp.down_proj = wrap_linear(layer.mlp.down_proj)
    # need to do this simultaneously with lm_head and embedding
    # model.lm_head = wrap_linear(model.lm_head)
    return model

def get_dora_transpose_model(model):
    class LinearWithTransposeDora(torch.nn.Module):
        def __init__(self, linear, rank):
            super().__init__()
            assert linear.bias is None
            W = linear.weight.T
            mag = torch.linalg.norm(W, dim=1, keepdim=True)
            self.W = torch.nn.Parameter(W / mag, requires_grad = False)
            self.mag = torch.nn.Parameter(mag)
            self.A = torch.nn.Parameter(torch.randn(linear.in_features, rank, dtype = torch.bfloat16)/np.sqrt(rank))
            self.B = torch.nn.Parameter(torch.zeros(rank, linear.out_features, dtype = torch.bfloat16 ))
        def forward(self, x):
            norm = torch.linalg.norm(self.W + 16 * self.A @ self.B, dim=1, keepdim=True)
            x *= (self.mag / norm).view(-1)
            return x @ self.W + 16 * x @ self.A @ self.B
    wrap_linear = partial(LinearWithTransposeDora, rank=8)
    for param in model.parameters():
        param.requires_grad = False
    for layer in model.model.layers:
        layer.self_attn.q_proj = wrap_linear(layer.self_attn.q_proj)
        layer.self_attn.k_proj = wrap_linear(layer.self_attn.k_proj)
        layer.self_attn.v_proj = wrap_linear(layer.self_attn.v_proj)
        layer.self_attn.o_proj = wrap_linear(layer.self_attn.o_proj)
        layer.mlp.gate_proj = wrap_linear(layer.mlp.gate_proj)
        layer.mlp.up_proj = wrap_linear(layer.mlp.up_proj)
        layer.mlp.down_proj = wrap_linear(layer.mlp.down_proj)
    # need to do this simultaneously with lm_head and embedding
    # model.lm_head = wrap_linear(model.lm_head)
    return model

def get_simple_dora_transpose_model(model):
    class LinearWithSimpleDoraTranspose(torch.nn.Module):
        def __init__(self, linear, rank):
            super().__init__()
            assert linear.bias is None
            W = linear.weight.T
            mag = torch.linalg.norm(W, dim=1, keepdim=True)
            self.W = torch.nn.Parameter(W / mag, requires_grad = False)
            self.mag = torch.nn.Parameter(mag)
            self.A = torch.nn.Parameter(torch.randn(linear.in_features, rank, dtype = torch.bfloat16)/np.sqrt(rank))
            self.B = torch.nn.Parameter(torch.zeros(rank, linear.out_features, dtype = torch.bfloat16 ))
        def forward(self, x):
            x *= self.mag.view(-1)
            return x @ self.W + 16 * x @ self.A @ self.B
    wrap_linear = partial(LinearWithSimpleDoraTranspose, rank=8)
    for param in model.parameters():
        param.requires_grad = False
    for layer in model.model.layers:
        layer.self_attn.q_proj = wrap_linear(layer.self_attn.q_proj)
        layer.self_attn.k_proj = wrap_linear(layer.self_attn.k_proj)
        layer.self_attn.v_proj = wrap_linear(layer.self_attn.v_proj)
        layer.self_attn.o_proj = wrap_linear(layer.self_attn.o_proj)
        layer.mlp.gate_proj = wrap_linear(layer.mlp.gate_proj)
        layer.mlp.up_proj = wrap_linear(layer.mlp.up_proj)
        layer.mlp.down_proj = wrap_linear(layer.mlp.down_proj)
    # need to do this simultaneously with lm_head and embedding
    # model.lm_head = wrap_linear(model.lm_head)
    return model

def get_simple_svdora_model(model):
    class LinearWithSimpleSvdora(torch.nn.Module):
        def __init__(self, linear, rank):
            super().__init__()
            assert linear.bias is None
            W = linear.weight.T
            print('here', W.shape)
            U, sigma, Vh = torch.linalg.svd(W.to(torch.float32), full_matrices=False)
            self.U = torch.nn.Parameter(U.to(W.dtype), requires_grad=False)
            self.sigma = torch.nn.Parameter(sigma.to(W.dtype))
            self.Vh = torch.nn.Parameter(Vh.to(W.dtype), requires_grad=False)
            print(self.sigma)
            
            self.A1 = torch.nn.Parameter(torch.randn(U.shape[0], rank, dtype = torch.bfloat16)/np.sqrt(rank))
            self.B1 = torch.nn.Parameter(torch.zeros(rank, U.shape[1], dtype = torch.bfloat16 ))
            self.A2 = torch.nn.Parameter(torch.randn(Vh.shape[0], rank, dtype = torch.bfloat16)/np.sqrt(rank))
            self.B2 = torch.nn.Parameter(torch.zeros(rank, Vh.shape[1], dtype = torch.bfloat16 ))
        def forward(self, x):
            x = x @ self.U + 16 * x @ self.A1 @ self.B1
            x = x * self.sigma
            return x @ self.Vh + 16 * x @ self.A2 @ self.B2
    wrap_linear = partial(LinearWithSimpleSvdora, rank=8)
    for param in model.parameters():
        param.requires_grad = False
    for layer in model.model.layers:
        layer.self_attn.q_proj = wrap_linear(layer.self_attn.q_proj)
        layer.self_attn.k_proj = wrap_linear(layer.self_attn.k_proj)
        layer.self_attn.v_proj = wrap_linear(layer.self_attn.v_proj)
        layer.self_attn.o_proj = wrap_linear(layer.self_attn.o_proj)
        layer.mlp.gate_proj = wrap_linear(layer.mlp.gate_proj)
        layer.mlp.up_proj = wrap_linear(layer.mlp.up_proj)
        layer.mlp.down_proj = wrap_linear(layer.mlp.down_proj)
    # need to do this simultaneously with lm_head and embedding
    # model.lm_head = wrap_linear(model.lm_head)
    return model

def train_lora():
    # model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype='auto')
    print(model)
    model_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total trainable parameters in model : {model_params}")

    lora_model = get_simple_svdora_model(model)
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
    

