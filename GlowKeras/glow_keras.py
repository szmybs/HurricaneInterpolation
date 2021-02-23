import numpy as np
import os
import math
import time
import sys

from PIL import Image

import tensorflow as tf
import keras.backend.tensorflow_backend as KTF

config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.8
session = tf.Session(config=config)
KTF.set_session(session)

from keras import backend as K
from keras.layers import Conv2D, Activation, Input, Lambda
from keras.models import Model
from keras.callbacks import Callback, ModelCheckpoint
from keras.optimizers import Adam

if __name__ == "__main__":
    sys.path.append(os.getcwd())

from GlowKeras.flow_layers import *

from DataSet.hurricane_generator import name_visibility_date_dir_generator, name_visibility_date_dir_data_counts
from DataSet.extract import HurricaneExtraction
from DataSet.normalize import Normalize, Quantization
from GlowKeras.data_augmentation import rotation90, vertical_flip, mirror_flip


gaussian_norm_path = "./DataSet/norm_factor/gaussian.npz"
max_min_norm_path = "./DataSet/norm_factor/max_min.npz"


class Evaluate(Callback):
    def __init__(self, drp, smp, srp, sample, reconstrut):
        self.lowest = 1e10
        self.drp , self.smp, self.srp = drp, smp, srp
        self.sample = sample
        self.reconstrut = reconstrut

    def on_epoch_end(self, epoch, logs=None):
        if (epoch+1) % 10 == 0:
            self.sample(save_path=self.srp, epoch=(epoch+1))
            self.reconstrut(data_root_path=self.drp, save_path=self.srp, epoch=(epoch+1))

        if logs['loss'] <= self.lowest:
            self.lowest = logs['loss']
            self.model.save_weights(os.path.join(self.smp,'best_encoder.weights'))
        elif logs['loss'] > 0 and epoch > 15:
            """在后面，loss一般为负数，一旦重新变成正数，就意味着模型已经崩溃，需要降低学习率。"""
            self.model.load_weights(os.path.join(self.smp,'best_encoder.weights'))
            K.set_value(self.model.optimizer.lr, K.get_value(self.model.optimizer.lr) * 0.1)


# 默认参数训练的模型为Model-base
class Glow(object):
    def __init__(self, batch_size=8, 
                                    lr=1e-4, 
                                    level=3, 
                                    depth=8, 
                                    vd='LIDIA', 
                                    data_shape=(256,256,5), 
                                    dam=[rotation90, vertical_flip, mirror_flip]
    ):
        self.batch_size = batch_size
        self.learning_rate = lr

        self.level = level
        self.depth = depth

        self.validate_seperation = vd

        self.data_shape = data_shape

        self.dam = dam


    def build_basic_model(self, in_channel):
        """基础模型，即耦合层中的模型（basic model for Coupling）
        """
        _in = Input(shape=(None, None, in_channel))
        _ = _in
        hidden_dim = 128        # 512->128
        _ = Conv2D(hidden_dim, (3, 3), padding='same')(_)
        # _ = Actnorm(add_logdet_to_loss=False)(_)
        _ = Activation('relu')(_)
        _ = Conv2D(hidden_dim, (1, 1), padding='same')(_)
        # _ = Actnorm(add_logdet_to_loss=False)(_)
        _ = Activation('relu')(_)
        _ = Conv2D(in_channel, (3, 3), kernel_initializer='zeros', padding='same')(_)

        return Model(_in, _)
    

    def compile(self):
        optimizer = Adam(lr=self.learning_rate, decay=1e-5)
        self.encoder.compile(loss=lambda y_true,y_pred: 0.5 * K.sum(y_pred**2, 1) + 0.5 * np.log(2*np.pi) * K.int_shape(y_pred)[1],
                optimizer=optimizer)
        

    def build_model(self, data_shape):
        # encoder part
        squeeze = Squeeze()

        inner_layers = []
        outer_layers = []
        for i in range(5):
            inner_layers.append([])

        for i in range(3):
            outer_layers.append([])

        x_in = Input(shape=data_shape)
        x = x_in
        x_outs = []

        x = Lambda(lambda s: K.in_train_phase(s + 1./256 * K.random_uniform(K.shape(s)), s))(x)

        for i in range(self.level):
            x = squeeze(x)
            for j in range(self.depth):
                actnorm = Actnorm()
                permute = Permute(mode='random')
                split = Split()
                couple = CoupleWrapper(self.build_basic_model(self.data_shape[-1]*2**(i+1))) 
                concat = Concat()
                inner_layers[0].append(actnorm)
                inner_layers[1].append(permute)
                inner_layers[2].append(split)
                inner_layers[3].append(couple)
                inner_layers[4].append(concat)
                x = actnorm(x)
                x = permute(x)
                x1, x2 = split(x)
                x1, x2 = couple([x1, x2])
                x = concat([x1, x2])
            if i < self.level-1:
                split = Split()
                condactnorm = CondActnorm()
                reshape = Reshape()
                outer_layers[0].append(split)
                outer_layers[1].append(condactnorm)
                outer_layers[2].append(reshape)
                x1, x2 = split(x)
                x_out = condactnorm([x2, x1])
                x_out = reshape(x_out)
                x_outs.append(x_out)
                x = x1
            else:
                for _ in outer_layers:
                    _.append(None)

        final_actnorm = Actnorm()
        final_concat = Concat()
        final_reshape = Reshape()

        x = final_actnorm(x)
        x = final_reshape(x)
        x = final_concat(x_outs+[x])

        # 加一个sigmoid试试伪装成截断分布
        # x = Lambda( lambda y: 2*K.sigmoid(y)-1 )(x)

        self.encoder = Model(x_in, x)
        for l in self.encoder.layers:
            if hasattr(l, 'logdet'):
                self.encoder.add_loss(l.logdet)

        self.encoder.summary()


        # decoder part
        x_in = Input(shape=K.int_shape(self.encoder.outputs[0])[1:])
        x = x_in

        # 从伪装的截断分布恢复
        # 该函数的导数 (1-y) / (1+y) 会出现nan的爆炸
        # x = Lambda( lambda y: K.log( (1+y) / (1-y) ) )(x)

        x = final_concat.inverse()(x)
        outputs = x[:-1]
        x = x[-1]
        x = final_reshape.inverse()(x)
        x = final_actnorm.inverse()(x)
        x1 = x

        for i,(split,condactnorm,reshape) in enumerate(list(zip(*outer_layers))[::-1]):
            if i > 0:
                x1 = x
                x_out = outputs[-i]
                x_out = reshape.inverse()(x_out)
                x2 = condactnorm.inverse()([x_out, x1])
                x = split.inverse()([x1, x2])
            for j,(actnorm,permute,split,couple,concat) in enumerate(list(zip(*inner_layers))[::-1][i*self.depth: (i+1)*self.depth]):
                x1, x2 = concat.inverse()(x)
                x1, x2 = couple.inverse()([x1, x2])
                x = split.inverse()([x1, x2])
                x = permute.inverse()(x)
                x = actnorm.inverse()(x)
            x = squeeze.inverse()(x)

        self.decoder = Model(x_in, x)

        self.decoder.summary()


    def nparray_to_image(self, x, save_path):
        height, width, channel = self.data_shape

        figure = np.zeros( (height * len(x), width * channel, 1) )

        for i in range(len(x)):
            for j in range(channel):
                figure[i*height : (i+1)*height, j*width : (j+1)*width] = x[i][..., j : (j+1)]
        
        figure = np.squeeze(figure)

        figure = np.clip(figure * 255, 0, 255).astype('uint8')      # 将0-1变为0-255，似乎很多软件只能打开0-255式的
        img = Image.fromarray(figure)

        # img = Image.fromarray(figure)

        postfix = os.path.splitext(save_path)[1]
        if postfix.upper() == '.JPG' or postfix.upper() == '.PNG':
            img = img.convert("RGB")
        
        # img.show()
        img .save(save_path)


    def sample(self, save_path, epoch, n=9, std=0.9):
        x = []

        for _ in range(n):
            decoder_input_shape = (1,) + K.int_shape(self.decoder.inputs[0])[1:]
            z_sample = np.array(np.random.randn(*decoder_input_shape)) * std
            x_decoded = self.decoder.predict(z_sample)
            digit = x_decoded[0].reshape(self.data_shape)
            digit = self.hurricane_undo_normalize(digit)   # 在这里需要一个undo_normalize
            x.append(digit)
        save_name = str(epoch) + "_sampled" + ".jpg"
        self.nparray_to_image(x, os.path.join(save_path, save_name))

    # 没什么用
    def reconstrut(self, data_root_path, save_path, epoch):
        x = []

        validate_data_generator = self.glow_generator(data_root_path, mode='validate')
        for _ in range(9):
            vd = next(validate_data_generator)
            lv = self.encoder.predict(vd[0])
            re = self.decoder.predict(lv)
            re = self.hurricane_undo_normalize(re)   # 在这里需要一个undo_normalize
            x.append(re)
        save_name = str(epoch) + ".jpg"
        self.nparray_to_image(x, os.path.join(save_path, save_name))


    # 在 hurrica_generator.py中浮点化和正则化
    def glow_generator(self, data_root_path, mode = 'train'):
        if mode == 'train':
            gg = name_visibility_date_dir_generator(root_path=data_root_path, 
                                        batch_size=self.batch_size, 
                                        wbl=[ {'black_list':[self.validate_seperation]}, {'white_list':['Visible']}, {} ])
        elif mode == 'validate':
            gg = name_visibility_date_dir_generator(root_path=data_root_path, 
                                        wbl=[ {'white_list':[self.validate_seperation]}, {'white_list':['Visible']}, {} ])
        else:
            gg = name_visibility_date_dir_generator(root_path=data_root_path, 
                                        batch_size=self.batch_size, 
                                        wbl=[ {}, {'white_list':['Visible']}, {} ])
        
        while True:
            gdx = next(gg)
            #在这里正则化
            gdx = self.hurricane_normalize(gdx)

            gadx = [gdx]
            if mode == 'train':
                for func in self.dam:
                    gadx.append(func(gdx))

            for dx in gadx:
                dy = dx.reshape( dx.shape[0], -1) 
                yield(dx, dy)


    def create_norm_list(self, data_root_path, gaussian_path=None, max_min_path=None):
        if hasattr(self, 'normalize'):
            return
        
        self.normalize = Normalize(data_path=data_root_path, gaussian_path=gaussian_path, max_min_path=max_min_path)

        self.norm_list = [self.normalize.normalize_using_max_min]
        self.undo_norm_list = [self.normalize.undo_normalize_using_max_min]

    def hurricane_normalize(self, data):
        if hasattr(self, 'norm_list') == False or len(self.norm_list) <= 0:
            return data           
        for norm in self.norm_list:
            data = norm(data)
        return data

    def hurricane_undo_normalize(self, data):
        if hasattr(self, 'undo_norm_list') == False or len(self.undo_norm_list) <= 0:
            return data  
        for unnorm in self.undo_norm_list:
            data = unnorm(data)
        return data


    def train(self, epochs, sample_interval=5, **kwarg):
        data_root_path = kwarg['drp']
        save_model_path = kwarg['smp']
        sample_root_path = kwarg['srp']

        # model
        self.build_model(self.data_shape)
        self.compile()

        self.create_norm_list(data_root_path, gaussian_norm_path, max_min_norm_path)

        # generator
        train_data_generator = self.glow_generator(data_root_path)
        validate_data_generator = self.glow_generator(data_root_path, mode='validate')

        # epoch
        train_data_nums = name_visibility_date_dir_data_counts(data_root_path, black_list=[self.validate_seperation, 'Invisible'])
        validate_data_nums = name_visibility_date_dir_data_counts(data_root_path, black_list=['Invisible']) - train_data_nums

        steps_per_epoch = math.floor( train_data_nums / self.batch_size ) * (len(self.dam) + 1)
        steps_for_validate = math.floor( validate_data_nums / self.batch_size )
        
        # checkpoint
        checkpoint_file_name = "weights.{epoch:02d}-{val_loss:.2f}.hdf5"
        checkpoint = ModelCheckpoint(os.path.join(save_model_path, checkpoint_file_name), 
                        monitor='val_loss', 
                        save_weights_only=False, 
                        verbose=1, 
                        save_best_only=False, 
                        period=sample_interval)

        # evaluate
        evaluate = Evaluate(data_root_path, save_model_path, sample_root_path, self.sample, self.reconstrut)

        # fit_generator
        try:
            trainingDuration = 0.0
            trainStartTime = time.time()
            history = self.encoder.fit_generator(
                        generator = train_data_generator,
                        steps_per_epoch = steps_per_epoch,
                        epochs = epochs,
                        verbose = 1,
                        callbacks = [checkpoint, evaluate],
                        validation_data = validate_data_generator,
                        validation_steps = steps_for_validate,
                        class_weight = None,
                        workers=1,
                        initial_epoch=0)
            trainingDuration = time.time() - trainStartTime
        except KeyboardInterrupt:
            print("Training duration (s): {}\nInterrupted by user!".format(trainingDuration))
        print("Training duration (s): {}".format(trainingDuration))


    def load_weights(self, weights_path):
        if hasattr(self, 'encoder') == False or hasattr(self, 'decoder') == False:
            self.build_model(self.data_shape)
        
        self.encoder.load_weights(weights_path)


# 没什么用
def test_glow_factory():
    glow = Glow()
    glow.load_weights("./GlowKeras/Model/best_encoder.weights")
    glow.create_norm_list(None, gaussian_norm_path, max_min_norm_path)
    return glow


if __name__ == "__main__":
    data_root_path = "./DataSet/ScaledData256-RadM/"
    save_model_path = "./GlowKeras/Model/"
    sample_root_path = "./GlowKeras/Sample/"

    if os.path.exists(save_model_path) == False:
        os.mkdir(save_model_path)
    if os.path.exists(sample_root_path) == False:
        os.mkdir(sample_root_path)

    glow = Glow()
    glow.train(30, drp=data_root_path, smp=save_model_path, srp=sample_root_path)

    # gg = glow.glow_generator(data_root_path)
    # for i in range(10):
    #     gd = next(gg)
    #     print(gd.shape)
