
from transformers import AutoModelForCausalLM, TrainingArguments, Trainer, AutoTokenizer
import torch
import datasets

from functools import partial
import time

MODEL_NAME = "google/gemma-2b"
DATASET = 'hackathon-pln-es/spanish-to-quechua'
SEQ_LEN = 20
OUT_DIR = 'data'

class LoRALayer(torch.nn.Module):
    def __init__(self, in_dim, out_dim, rank, alpha):
        super().__init__()
        std_dev = 1 / torch.sqrt(torch.tensor(rank).float())
        self.A = torch.nn.Parameter(torch.randn(in_dim, rank) * std_dev)
        self.B = torch.nn.Parameter(torch.zeros(rank, out_dim))
        self.alpha = alpha

    def forward(self, x):
        x = self.alpha * (x @ self.A @ self.B)
        return x

class LinearWithLoRA(torch.nn.Module):
    def __init__(self, linear, rank, alpha):
        super().__init__()
        self.linear = linear
        self.lora = LoRALayer(
            linear.in_features, linear.out_features, rank, alpha
        )

    def forward(self, x):
        return self.linear(x) + self.lora(x)
    

def get_lora_model(model):
    # default hyperparameter choices
    lora_r = 8
    lora_alpha = 16
    lora_dropout = 0.05
    lora_query = True
    lora_key = True
    lora_value = True
    lora_projection = True
    lora_mlp = False
    lora_head = False
  
    assign_lora = partial(LinearWithLoRA, rank=lora_r, alpha=lora_alpha)

    for param in model.parameters():
        param.requires_grad = False

    for layer in model.model.layers:
        if lora_query:
            layer.self_attn.q_proj = assign_lora(layer.self_attn.q_proj)
        if lora_key:
            layer.self_attn.k_proj = assign_lora(layer.self_attn.k_proj)
        if lora_value:
            layer.self_attn.v_proj = assign_lora(layer.self_attn.v_proj)
        if lora_projection:
            layer.self_attn.o_proj = assign_lora(layer.self_attn.o_proj)
        # if lora_mlp:
        #     layer.fc1 = assign_lora(layer.fc1)
        #     layer.fc2 = assign_lora(layer.fc2)

    if lora_head:
        model.model.lm_head = assign_lora(model.model.lm_head)

    return model

def train_lora():
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
    print(model)
    model_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total trainable parameters in model : {model_params}")

    lora_model = get_lora_model(model)
    print(lora_model)
    lora_model_params = sum(p.numel() for p in lora_model.parameters() if p.requires_grad)
    print(f"Total trainable parameters in lora model : {lora_model_params} and are {(lora_model_params/model_params)*100} % of the original model")
    
    lm_dataset = getDataset()
    train(lora_model, lm_dataset, OUT_DIR)
    

def train(model, lm_dataset, output_dir):
    
    training_args = TrainingArguments(
        output_dir=output_dir,
        eval_strategy="epoch",
        learning_rate=2e-5,
        weight_decay=0.01,
        torch_compile=True,
        eval_on_start=True,
        save_steps=10000
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=lm_dataset["train"],
        eval_dataset=lm_dataset["validation"],
        # data_collator=data_collator,
    )

    st = time.time()
    trainer.train()
    et = time.time()

    print(f"total training time : {(et - st)} sec.")


def getDataset():
    print(f'\nin getDataset')
    from datasets import load_dataset
    data = load_dataset(DATASET)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    print(data)

    #split data
    # data = data["train"].train_test_split(test_size=.2, seed=1)

    def preprocess(data_row, tokenizer):
        return tokenizer(data_row['qu'])

    data = data.map( preprocess,
                    # batched = True,
                    # num_proc = 4,
                    fn_kwargs = {'tokenizer' : tokenizer},
                    remove_columns = data['train'].column_names
                    )
    
    def group_texts(examples, block_size):
        # Concatenate all texts.
        concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
        # customize this part to your needs.

        # if total_length >= block_size:
        total_length = (total_length // block_size) * block_size
        
        # Split by chunks of block_size.
        result = {
            k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
            for k, t in concatenated_examples.items()
        }

        # labels because the model expects the argument to be named labels
        result["labels"] = result["input_ids"].copy()
        # del result['input_ids']
        return result


    lm_dataset = data.map(group_texts, 
                        batched=True,
                        num_proc=4,
                        fn_kwargs = {'block_size' : SEQ_LEN })
    
    print(lm_dataset['train'])
    print(lm_dataset['train'][0])

    return lm_dataset


if __name__ == '__main__':
    train_lora()
