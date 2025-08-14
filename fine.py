import transformers
import numpy as np
import torch
import jax
import jax.numpy as jnp
import numpy as np

# model_name = "EleutherAI/gpt-neo-125m"
# model_name = "EleutherAI/gpt-neo-1.3B"
# model_name = "EleutherAI/gpt-neo-2.7B" # too big
# model_name = "YALCINKAYA/opsgenius_s" # quantization of above
# model_name = "EleutherAI/gpt-j-6b" # too big
# model_name = "YALCINKAYA/opsgenius_j" # quantization of above
# model_name = "Qwen/Qwen3-4b-Base" # not flax transformers
# model_name = "google-bert/bert-base-cased"
model_name = "google/gemma-2b"
# model_name = "google/gemma-2-2b"

tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
model = transformers.AutoModelForCausalLM.from_pretrained(model_name,torch_dtype='auto')

# text = "The current president of the US is "
# torch_inputs = tokenizer([text]*1, return_tensors="pt")

B = 1
seq = 500
torch_inputs = { 'input_ids' : torch.randint(high=256000, size=(B, seq)),
                'attention_mask' : torch.ones((B,seq)) }

res1 = model.to('cuda')(**{k: v.to('cuda') for k,v in torch_inputs.items()})

# from torch2jax import t2j
# jax_inputs = {k: t2j(v) for k,v in torch_inputs.items()}
# state_dict = {k: t2j(v) for k,v in model.state_dict().items()}
# modelf = t2j(model)
# def logits(*args,state_dict=None):
#     res = modelf(*args, state_dict=state_dict)
#     return res.logits
# res = jax.jit(logits,backend='gpu')(*[jax_inputs[k] for k in ['input_ids', 'attention_mask']],state_dict=state_dict)
# print(res)

# generated_ids = model.generate(
#     **jax_inputs,
#     max_new_tokens=100,
#     do_sample = True,
#     num_beams = 1
# )
# # generated_ids = np.array(generated_ids.sequences)
# response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)



# from datasets import load_dataset
# dataset = load_dataset("yelp_review_full")

# def tokenize(examples):
#     return tokenizer(examples["text"], padding="max_length", truncation=True)

# dataset = dataset.map(tokenize, batched=True)


# import evaluate
# import numpy as np

# metric = evaluate.load("accuracy")

# def compute_metrics(eval_pred):
#     logits, labels = eval_pred
#     # convert the logits to their predicted class
#     predictions = np.argmax(logits, axis=-1)
#     return metric.compute(predictions=predictions, references=labels)

# from transformers import TrainingArguments
# from transformers import Trainer

# training_args = TrainingArguments(
#     output_dir="data",
#     eval_strategy="epoch",
# )

# trainer = Trainer(
#     model=model,
#     args=training_args,
#     train_dataset=dataset["train"],
#     eval_dataset=dataset["test"],
#     compute_metrics=compute_metrics,
# )
# trainer.train()
