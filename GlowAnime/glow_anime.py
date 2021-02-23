import numpy as np
import os
import math
import time
import sys
import random
import glob

from PIL import Image

from keras import backend as K
from keras.callbacks import ModelCheckpoint

import tensorflow as tf
import keras.backend.tensorflow_backend as KTF

config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.8
session = tf.Session(config=config)
KTF.set_session(session)


if __name__ == "__main__":
    sys.path.append(os.getcwd())

from GlowKeras.glow_keras import Glow


class NewEvaluate(object):
    def __init__(self):
        self.lowest = 1e10
    
    def on_epoch_end(self, epoch, model, loss, best_weights_path):
        if loss <= self.lowest:
            self.lowest = loss
            model.save_weights(best_weights_path)
        elif (np.isnan(loss) or loss > 0) and os.path.exists(best_weights_path):
            model.load_weights(best_weights_path)
            K.set_value(model.optimizer.lr, K.get_value(model.optimizer.lr) * 0.1)
            print("在第%d个epoch参数回溯" % (epoch+1))


class GlowAnime(Glow):
    def __init__(self, batch_size=32, 
                                    lr=1e-3, 
                                    level=3, 
                                    depth=10, 
                                    vd='validate_data', 
                                    data_shape=(64,64,3), 
                                    dam=None
    ):
        super(GlowAnime, self).__init__(
            batch_size,
            lr,
            level,
            depth,
            vd,
            data_shape,
            dam
        )


    def anime_generator(self, data_root_path):
        imgs = os.listdir(data_root_path)

        while True:
            random.shuffle(imgs)       
            batch_im = []
            for img in imgs:
                im = Image.open(os.path.join(data_root_path, img))
                im = np.asarray(im)
                batch_im.append(im)
                if len(batch_im) == self.batch_size:
                    yield(self.norm(np.asarray(batch_im)))
                    batch_im.clear()


    def norm(self, data):
        return data / 255 * 2 - 1
    
    def undo_norm(self, data):
        data = (data + 1) / 2 * 255
        data = np.clip(data, 0, 255).astype('uint8')
        return data


    def train(self, epochs, sample_interval=5, bp=None, **kwarg):
        train_data_path = kwarg['drp']
        save_model_path = kwarg['smp']
        sample_root_path = kwarg['srp']

        # model
        self.build_model(self.data_shape)
        self.compile()

        if bp != None:
            self.encoder.load_weights(bp)

        # generator
        train_data_generator = self.anime_generator(train_data_path)

        # epoch
        train_data_nums = len(os.listdir(train_data_path))

        steps_per_epoch = math.floor( train_data_nums / self.batch_size )
        
        evaluate = NewEvaluate()
        # fit_generator
        try:
            for epoch in range(epochs):
                total_loss = 0
                for step in range(steps_per_epoch):
                    trainingDuration = 0.0
                    trainStartTime = time.time()

                    img = train_data_generator.__next__()

                    # lc = self.encoder.predict(img)
                    clr = K.get_value(self.encoder.optimizer.lr)

                    # def get_weights():
                    #     actnorm_weights = []
                    #     for i in range(49):
                    #         layer_name = "actnorm_" + str(i+1)
                    #         weights = self.encoder.get_layer(layer_name).get_weights()
                    #         actnorm_weights.append(weights)

                    #     cond_actnorm_weights = []
                    #     for i in range(2):
                    #         layer_name = "cond_actnorm_" + str(i+1)
                    #         weights = self.encoder.get_layer(layer_name).get_weights()
                    #         cond_actnorm_weights.append(weights)

                    #     permute_weights = self.encoder.get_layer("permute_24").get_weights()
                    #     model_weights = self.encoder.get_layer("model_19").get_weights()
                    #     print()
                    # get_weights()

                    # max_weights = []
                    # for layer in self.encoder.layers:
                    #     w = layer.get_weights()
                    #     max_weights.append(w)

                    loss = self.encoder.train_on_batch(x=img, y=img.reshape( img.shape[0], -1))
                    trainingDuration = time.time() - trainStartTime

                    total_loss = total_loss + loss
                    mean_loss = total_loss / (step + 1)
                    print("%d  -  time : %f  -  %d - [loss: %f] - [ml: %f]" % (epoch+1, trainingDuration, step, loss, mean_loss) , end='\r')
                print()

                evaluate.on_epoch_end(epoch, self.encoder, total_loss, os.path.join(save_model_path, "best_weights.hdf5"))
                # time.sleep(2)

                if (epoch+1) % 100 == 0:
                    K.set_value(self.encoder.optimizer.lr, K.get_value(self.encoder.optimizer.lr) * 0.1)

                if (epoch+1) % sample_interval == 0:
                    self.sample(sample_root_path, epoch+1)

                    save_name = "weights_" + str(epoch+1) + '_' + str(round(total_loss / steps_per_epoch)) + ".hdf5"
                    self.encoder.save_weights(os.path.join(save_model_path, save_name))

        except KeyboardInterrupt:
            print("Training duration (s): {}\nInterrupted by user!".format(trainingDuration))
        print("Training duration (s): {}".format(trainingDuration))


    def sample(self, save_path, epoch, n=9, std=0.9):
        height, width, channel = self.data_shape
        figure = np.zeros( (height * n, width * n, channel) )

        for i in range(n):
            for j in range(n):
                decoder_input_shape = (1,) + K.int_shape(self.decoder.inputs[0])[1:]
                z_sample = np.array(np.random.randn(*decoder_input_shape)) * std
                x_decoded = self.decoder.predict(z_sample)
                digit = x_decoded[0].reshape(self.data_shape)

                figure[i*height : (i+1)*height, j*width : (j+1)*width] = digit

        figure = self.undo_norm(figure)
        figure = Image.fromarray(figure)

        save_name = str(epoch) + "_sampled" + ".jpg"
        figure.save(os.path.join(save_path, save_name))



if __name__ == "__main__":
    data_root_path = "/home/my/Download/Anime_face_dataset/train_data/"
    save_model_path = "./GlowAnime/Model/"
    sample_root_path = "./GlowAnime/Sample/"

    if os.path.exists(save_model_path) == False:
        os.mkdir(save_model_path)
    if os.path.exists(sample_root_path) == False:
        os.mkdir(sample_root_path)

    glow = GlowAnime()
    # glow.train(500, 1, "./GlowAnime/Model/best_weights.hdf5", drp=data_root_path, smp=save_model_path, srp=sample_root_path)
    glow.train(500, 5, None, drp=data_root_path, smp=save_model_path, srp=sample_root_path)