
from transformers import AutoModelForCausalLM, TrainingArguments, Trainer, AutoTokenizer
import torch
import datasets
import numpy as np
import jax
import jax.numpy as jnp
import optax
from torch2jax import t2j
import functools
from tqdm import tqdm
import operator

import peft

import os
os.environ['XLA_FLAGS'] = (
    '--xla_gpu_triton_gemm_any=True '
    '--xla_gpu_enable_latency_hiding_scheduler=true '
)

MODEL_NAME = "google/gemma-2b"
# MODEL_NAME = "NousResearch/Llama-3.2-1B"
SEQ_LEN = 463
OUT_DIR = 'data'

def train_peft():
    # model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype='auto')
    model_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total trainable parameters in model : {model_params}")

    # peft_model = peft.get_lora_model(model)
    # peft_model = peft.get_simple_dora_model(model)
    # peft_model = peft.get_dora_model(model)
    # peft_model = peft.get_tied_lora_model(model,r=16)
    # peft_model = peft.get_tied_lora_extra_model(model,a=16,b=16)
    # peft_model = peft.get_simple_dora_model(model)
    peft_model = peft.get_tensor_contraction_model(model,a=16,b=16,l=8,premult=False,postmult=False)
    
    # peft_model = model
    # peft_model.lm_head.requires_grad = False
    # peft_model.model.embed_tokens.requires_grad = False
    
    peft_model_params = sum(p.numel() for p in peft_model.parameters() if p.requires_grad)
    print(f"Total trainable parameters in peft model : {peft_model_params} and are {(peft_model_params/model_params)*100} % of the original model")
    
    dataset = get_dataset_gsm8k()
    # train(peft_model, dataset, OUT_DIR)
    train_jax(peft_model, dataset, OUT_DIR)
    
class ModelWithLoss(torch.nn.Module): 
    def __init__(self, model):
        super().__init__()
        self.model = model
        # self.loss = torch.nn.CrossEntropyLoss(reduction="none")
        # self.loss = torch.nn.CrossEntropyLoss(reduction="sum")
        self.loss = torch.nn.CrossEntropyLoss()
    def forward(self, *, input_ids=None, attention_mask=None ):
        assert input_ids is not None
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
        learning_rate=2.5e-6/4,
        weight_decay=0.01,
        # torch_compile=True,
        save_strategy='no',
        per_device_train_batch_size = 1,
        per_device_eval_batch_size = 1,
        save_safetensors = False, # work around bug 
        # bf16=True,
        gradient_accumulation_steps=1,
        logging_steps=500,
        # eval_on_start=True
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

def train_jax(model_torch, lm_dataset, output_dir):
    epochs = 3
    batchsize = 1
    seed = 0
    logging_steps = 250
    # start_learning_rate = 0.1
    # start_learning_rate = 1.5e-5 # lora
    start_learning_rate = 1.5e-5/2
    weight_decay = 0.001

    key = jax.random.key(seed)
    
    state_dict = model_torch.state_dict(keep_vars=True)
    trainable_keys = {}
    nontrainable_keys = {}
    for k,v in state_dict.items():
        assert v.is_contiguous()
        if v.requires_grad:
            assert v.data_ptr() not in nontrainable_keys, "aliased parameters must agree whether they requires_grad"
            trainable_keys.setdefault(v.data_ptr(), []).append(k)
        else:
            assert v.data_ptr() not in trainable_keys, "aliased parameters must agree whether they requires_grad"
            nontrainable_keys.setdefault(v.data_ptr(), []).append(k)
            
    trainable_state_dict = { tuple(ks) : t2j(state_dict[ks[0]]) for ks in trainable_keys.values() }
    nontrainable_state_dict = { tuple(ks) : t2j(state_dict[ks[0]]) for ks in nontrainable_keys.values() }
    model_jax = t2j(model_torch)
    del model_torch, state_dict, trainable_keys, nontrainable_keys
    def model(input_ids, trainable_state_dict, nontrainable_state_dict):
        state_dict = dict({k: v for ks, v in trainable_state_dict.items() for k in ks}, 
                          **{k: v for ks, v in nontrainable_state_dict.items() for k in ks})
        return model_jax(input_ids, state_dict = state_dict)
    
    def get_batches(key, split='train'):
        from itertools import batched
        split = list(lm_dataset[split])
        maxex = max([len(ex['input_ids']) for ex in split])
        # bs = [1]
        # bs = [1,2]
        bs = [1,2,3,4]
        split_by_b = { }
        for ex in split:
            b = next(b for b in reversed(bs) if len(ex['input_ids']) <= maxex/b**0.9)
            split_by_b.setdefault(b,[]).append(ex)
        print(f'{len(split_by_b)} groups of batches by example length')
        print(sorted([(b,len(exs),max(len(ex['input_ids']) for ex in exs)) for b, exs in split_by_b.items()])) 
        batches = []
        key, keycur = jax.random.split(key)
        for (b, exs), keycur in zip(sorted(split_by_b.items()),jax.random.split(keycur,len(split_by_b))):
            # seq_len = int(np.floor(maxex / np.sqrt(b)))
            seq_len = max(len(ex['input_ids']) for ex in exs)
            # b = max(b-1,1)
            b *= batchsize
            ixs = jax.random.permutation(keycur, len(exs))
            for batch in batched(ixs, b):
                batch = [exs[i] for i in batch]
                input_ids = np.zeros((b, seq_len), dtype=jnp.int32)
                id_mask = np.zeros((b, seq_len), dtype=jnp.bool)
                for i,ex in enumerate(batch):
                    ids = ex['input_ids']
                    input_ids[i, :len(ids)] = ids
                    id_mask[i, :len(ids)] = True
                batches.append((input_ids, id_mask))
        ixs = jax.random.permutation(key,len(batches))
        return [batches[i] for i in ixs]
    

            
    @jax.jit
    def loss_fn(trainable_state_dict,nontrainable_state_dict,input_ids,id_mask):
        result = model(input_ids, trainable_state_dict, nontrainable_state_dict)
        logits = result.logits[...,:-1,:]
        id_mask = id_mask[...,1:]
        labels = input_ids[...,1:]
        cross_entropy = optax.losses.softmax_cross_entropy_with_integer_labels(
                logits.reshape(-1, logits.shape[-1]), labels.reshape(-1),
                )
        return jnp.sum(jnp.where(id_mask.reshape(-1), cross_entropy, 0.0)), jnp.sum(id_mask)
    
    @functools.partial(jax.jit,donate_argnums=[0,1])
    def update_function(trainable_state_dict, opt_state, nontrainable_state_dict, input_ids, id_mask):
        (loss, tokens), grads = jax.value_and_grad(loss_fn,has_aux=True)(
                trainable_state_dict,nontrainable_state_dict,input_ids,id_mask)
        grad_norm_square = jnp.sqrt(jax.tree.reduce(operator.add, jax.tree.map(lambda e: jnp.sum(e**2), grads)))
        updates, opt_state = optimizer.update(grads, opt_state, trainable_state_dict)
        trainable_state_dict = optax.apply_updates(trainable_state_dict, updates)
        return loss, tokens, grad_norm_square, trainable_state_dict, opt_state

    test_batches = get_batches(jax.random.key(0), split='test')
    def evaluate(trainable_state_dict, nontrainable_state_dict):
        loss = 0.0
        num = 0
        for batch in tqdm(test_batches):
            cur_loss, tokens = loss_fn(trainable_state_dict, nontrainable_state_dict, *batch)
            loss += float(cur_loss)
            num += tokens
        return loss / num
    
    # eval_loss = evaluate(trainable_state_dict, nontrainable_state_dict)
    # print(f'eval_loss = {float(eval_loss)}')
    
    batches = [batch for key in jax.random.split(key, epochs) for batch in get_batches(key)]
    epoch_its = len(batches) // epochs
    
    schedule = optax.schedules.warmup_cosine_decay_schedule(start_learning_rate/10, start_learning_rate, 
                                                            warmup_steps = len(batches) // 5, decay_steps=len(batches))
    # schedule = optax.schedules.cosine_decay_schedule(start_learning_rate, decay_steps=len(batches))
    # optimizer = optax.adam(schedule)
    # optimizer = optax.adamw(schedule)
    optimizer = optax.adamw(schedule,weight_decay=weight_decay)
    
    opt_state = optimizer.init(trainable_state_dict)

    cum_loss = 0.0
    cum_grad_norm_square = 0.0
    n = 0
    for it,batch in tqdm(enumerate(batches),total=len(batches)):
        loss, tokens, grad_norm_square, trainable_state_dict, opt_state = update_function(
                trainable_state_dict, opt_state, nontrainable_state_dict, *batch
            )
        cum_loss += float(loss)
        cum_grad_norm_square += float(grad_norm_square)
        n += int(tokens)
        if it % logging_steps == logging_steps-1 or it == len(batches)-1:
            print({'loss' : cum_loss / n, 
                   'learning_rate' : float(schedule(it)), 
                   'grad_norm' : float(np.sqrt(cum_grad_norm_square / n)),
                   'epoch' : it / epoch_its })
            cum_loss = 0.0
            cum_grad_norm_square = 0.0
            n = 0
        if it % epoch_its == epoch_its-1 or it == len(batches)-1:
            eval_loss = evaluate(trainable_state_dict, nontrainable_state_dict)
            print(f'eval_loss = {float(eval_loss)}')
            
    return trainable_state_dict


def get_dataset_gsm8k(randomize=False):
    data = datasets.load_dataset('openai/gsm8k', 'main')
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    
    if randomize:
        data = datasets.concatenate_datasets([data['train'],data['test']])
        data = data.train_test_split(test_size=0.15)
        
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
            "source_lens": source_lens,
        }
    
    return data.map(preprocess, batched=True)

if __name__ == '__main__':
    trainer = train_peft()
