from transformers import ViTFeatureExtractor, ViTConfig, ViTModel

config = ViTConfig()

model = ViTModel(config)

config = model.config

data_set = 'autoencoder/dataset/'

