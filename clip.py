from transformers import FlaxCLIPModel

clip = FlaxCLIPModel.from_pretrained('openai/clip-vit-base-patch32')
# Note: FlaxCLIPModel is not a Flax Module
def load_model():
  clip = FlaxCLIPModel.from_pretrained('openai/clip-vit-base-patch32')
  module = clip.module # Extract the Flax Module
  variables = {'params': clip.params} # Extract the parameters
  return module, variables
