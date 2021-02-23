import numpy as np
import sys
import os
from PIL import Image
from keras.models import Model, load_model

if __name__ == "__main__":
    sys.path.append(os.getcwd())

from GlowAnime.glow_anime import GlowAnime

glow_anime = GlowAnime()
glow_anime.load_weights("./GlowAnime/Model/weights_2_-15065.0.hdf5")

glow_anime.encoder.summary()

actnorm_weights = []
for i in range(25):
    layer_name = "actnorm_" + str(i+1)
    weights = glow_anime.encoder.get_layer(layer_name).get_weights()
    actnorm_weights.append(weights)

cond_actnorm_weights = []
for i in range(2):
    layer_name = "cond_actnorm_" + str(i+1)
    weights = glow_anime.encoder.get_layer(layer_name).get_weights()
    cond_actnorm_weights.append(weights)

permute_weights = glow_anime.encoder.get_layer("permute_24").get_weights()
model_weights = glow_anime.encoder.get_layer("model_19").get_weights()

img = Image.open("/home/my/Download/Anime_face_dataset/train_data/1.png")
img = np.asarray(img)
img = np.expand_dims(glow_anime.norm(img), 0)
lc = glow_anime.encoder.predict(img)

print()