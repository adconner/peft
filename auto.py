import transformers
import jax
import jax.numpy as jnp

# model = transformers.FlaxAutoModelForSeq2SeqLM.from_pretrained('openai-community/gpt2')
# model = transformers.FlaxAutoModel.from_pretrained('openai-community/gpt2')

module = model.module 
params = model.params

