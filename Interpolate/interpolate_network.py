import numpy as np
import os
import math
import time
import sys
from abc import abstractmethod

from PIL import Image

import tensorflow as tf
import keras.backend.tensorflow_backend as KTF

config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.8
session = tf.Session(config=config)
KTF.set_session(session)

import keras.backend as K
import keras.initializers as initializers
from keras.layers import Input, Add, Flatten
from keras.layers.convolutional import Conv1D, Conv2D
from keras.models import Sequential, Model, save_model, load_model
from keras.layers import Layer
from keras.optimizers import Adam


if __name__ == "__main__":
    sys.path.append(os.getcwd())

from GlowKeras.glow_keras import Glow
from DataSet.hurricane_generator import name_visibility_date_dir_seq_generator, name_visibility_date_dir_data_counts
from GlowKeras.data_augmentation import rotation90, vertical_flip, mirror_flip
from GlowKeras.data_augmentation import rotation90_in_list, vertical_flip_in_list, mirror_flip_in_list
from DataSet.normalize import Normalize, Quantization

import Option


class InternalLayer(Layer):
    def __init__(self, parameter_initializer='ones', **kwargs):
        super(InternalLayer, self).__init__(**kwargs)
        self.parameter_initializer = initializers.get(parameter_initializer)
        self.parameter_regularizer = None
        self.parameter_constraint = None
        self.epsilon = None

    def build(self, input_shape):
        # shape = (input_shape[-1], )
        shape = input_shape[1:]
        self.parameter = self.add_weight(shape=shape,
                                        name='parameter',
                                        initializer=self.parameter_initializer,
                                        regularizer=self.parameter_regularizer,
                                        constraint=self.parameter_constraint,
                                        trainable=True)
        super(InternalLayer, self).build(input_shape)

    def call(self, inputs):
        # return tf.add( tf.multiply(inputs, self.parameter), inputs )
        return inputs * self.parameter + inputs

    def compute_output_shape(self, input_shape):
        return input_shape


class InterpolationBase(object):
    def __init__(self, 
                batch_size=8, 
                lr=1e-4, 
                data_shape=(256,256,5), 
                block_nums=5, 
                glow_model = None, 
                glow_inverse_model = None,
    ):
        self.batch_size = batch_size
        self.learning_rate = lr

        self.data_shape = data_shape
        self.block_nums = block_nums

        self.glow_model = glow_model
        self.glow_inverse_model = glow_inverse_model


    def build_model(self):
        flatten_size = 1
        for i in self.data_shape:
            flatten_size = flatten_size * i 

        inputs = Input(shape=(flatten_size, ), name='Input')
        outputs_img = []
        outputs_vec = []

        glow_inverse = self.glow_inverse_model
        inter = InternalLayer()

        glow_inverse.trainable = False

        # x = glow(inputs)
        # x = inter(x)
        # # x = glow_inverse(x)
        # outputs = x

        # x = glow(inputs)
        x = inputs
        for _ in range(self.block_nums-1):
            x = inter(x)
            y = glow_inverse(x)
            outputs_vec.append(x)
            outputs_img.append(y)
        outputs = outputs_img + outputs_vec

        return Model(inputs, outputs, name='interpolate_network')


    def pretreatment(self, data, batch_size):
        x = self.glow_model.predict(data[0], batch_size=batch_size)
        ground_truth = data[1]
        tmp = []
        for i in ground_truth:
            vec = self.glow_model.predict(i, batch_size=batch_size)
            tmp.append(vec)
        y = ground_truth + tmp
        return x, y


    def train(self, epochs=1, sample_interval=5, **kwarg):
        train_data_path = kwarg['tdp']
        validate_data_path = kwarg['vdp']
        save_model_path = kwarg['smp']
        interpolate_path = kwarg['itp']

        train_data_generator = self.generator(train_data_path, mode='train')
        validate_data_generator = self.generator(validate_data_path, mode='validate')

        steps_per_epoch, steps_per_validate = self.compute_step_per_epoch(train_data_path, validate_data_path)

        self.model = self.build_model()
        self.model.summary(line_length=150)
        self.compile()

        try:
            for epoch in range(epochs):
                total_loss = 0
                for step in range(steps_per_epoch):
                    trainStartTime = time.time()

                    td = train_data_generator.__next__()
                    
                    x, y = self.pretreatment(td, self.batch_size)
                    loss = self.model.train_on_batch(x=x, y=y)
                    # loss = self.model.train_on_batch(x=td[0], y=ground_truth[0])

                    total_loss = total_loss + loss[0]
                    mean_loss = total_loss / (step+1)
                    metric = loss[-1] + loss[-2]

                    trainingDuration = time.time() - trainStartTime
                    print("%d  -  time: %f  -  %d - [loss: %f] - [metric: %f]" % (epoch+1, trainingDuration, step+1, mean_loss, metric) , end='\r')
                print()

                self.validate(generator=validate_data_generator, steps=steps_per_validate)
                self.interpolate(generator=validate_data_generator, nums=9, save_path=interpolate_path, suffix_name=str(epoch+1))

                if (epoch+1) % sample_interval == 0:
                    save_name = "weights_" + str(epoch+1) + '_' + str(round(total_loss / steps_per_epoch, 3)) + ".hdf5"
                    self.save_weights(self.model, os.path.join(save_model_path, save_name))

        except KeyboardInterrupt:
            print("Interrupted by user!") 


    def validate(self, generator, steps):
        total_loss = 0
        for _ in range(steps):
            vd = generator.__next__()
            x, y = self.pretreatment(vd, 1)
            loss = self.model.test_on_batch(x=x, y=y)
            total_loss = total_loss + loss[0]
        mean_loss = total_loss / steps
        print("validate loss : %f" % (mean_loss))


    def interpolate(self, generator, nums, save_path, suffix_name):
        for i in range(nums):
            data = generator.__next__()
            x, y = self.pretreatment(data, 1)

            outputs = self.model.predict(x, batch_size=1)
            outputs_img = outputs[:self.block_nums]

            print("插值未完成")


    def save_weights(self, model, save_path):
        print("save_weights {}".format(save_path))
        model.save_weights(save_path)

    def load_weights(self, load_path):
        if hasattr(self, 'model') == False:
            self.model = self.build_model()
        print("load_weights {}".format(load_path))
        self.model.load_weights(load_path)

    def save_model(self, model, save_path):
        print("save_model {}".format(save_path))
        model.save(save_path)

    def load_model(self, load_path):
        print("load_model {}".format(load_path))
        self.model = load_model(load_model)


    @abstractmethod
    def compile(self):
        pass

    @abstractmethod
    def generator(self, data_root_path, mode='train'):
        pass
    
    @abstractmethod
    def compute_step_per_epoch(self, tdp, vdp):
        pass

    @abstractmethod
    def nparray_to_image(self, x, save_path):
        pass

    @abstractmethod
    def interpolate_normalize(self, data):
        pass

    @abstractmethod
    def interpolate_undo_normalize(self, data):
        pass


class HurricaneInterpolation(InterpolationBase):
    def __init__(self, 
                batch_size=8, 
                lr=1e-4, 
                data_shape=(256,256,5), 
                block_nums=5, 
                glow_model = None, 
                glow_inverse_model = None,
                vd='LIDIA', 
                dam=[rotation90_in_list, vertical_flip_in_list, mirror_flip_in_list]
    ):
        super(HurricaneInterpolation, self).__init__(batch_size, lr, data_shape, block_nums, glow_model, glow_inverse_model)
        self.validate_seperation = vd
        self.dam = dam


    def compile(self):
        optimizer = Adam(lr=self.learning_rate, decay=1e-5)
        img_loss = ['mse'] * (self.block_nums -1)
        vec_loss = ['mse'] * (self.block_nums - 1)
        loss = img_loss + vec_loss
        self.model.compile(optimizer=optimizer, loss=loss, metrics=['mae'])
        # self.model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])


    def generator(self, data_root_path, mode = 'train'):
        if mode == 'train':
            gg = name_visibility_date_dir_seq_generator(root_path=data_root_path, 
                                        batch_size=self.batch_size, 
                                        length=self.block_nums,
                                        wbl=[ {'black_list':[self.validate_seperation]}, {'white_list':['Visible']}, {} ])
        elif mode == 'validate':
            gg = name_visibility_date_dir_seq_generator(root_path=data_root_path, 
                                        batch_size=1,
                                        length=self.block_nums,
                                        wbl=[ {'white_list':[self.validate_seperation]}, {'white_list':['Visible']}, {} ])
        else:
            gg = name_visibility_date_dir_seq_generator(root_path=data_root_path, 
                                        batch_size=self.batch_size, 
                                        length=self.block_nums,
                                        wbl=[ {}, {'white_list':['Visible']}, {} ])
        while True:
            gdx = next(gg)
            #在这里正则化
            gdx = self.interpolate_normalize(gdx)

            gadx = []
            for g in gdx:
                gadx.append(g.astype(np.float32))
            gadx = [gadx]

            if mode == 'train':
                for func in self.dam:
                    gadx.append(func(gdx))

            # for dx in gadx:
            #     dy = dx.reshape( dx.shape[0], -1) 
            #     yield(dx, dy)
            for dx in gadx:
                yield(dx[0], dx[1:])



    def create_norm_list(self, data_root_path, gaussian_path=None, max_min_path=None):
        if hasattr(self, 'normalize'):
            return
        self.normalize = Normalize(data_path=data_root_path, gaussian_path=gaussian_path, max_min_path=max_min_path)

        self.norm_list = [self.normalize.normalize_using_max_min]
        self.undo_norm_list = [self.normalize.undo_normalize_using_max_min]

    def interpolate_normalize(self, data):
        if hasattr(self, 'norm_list') == False or len(self.norm_list) <= 0:
            return data

        if isinstance(data, list):
            for d in data:
                for norm in self.norm_list:
                    d = norm(d)
        else:  
            for norm in self.norm_list:
                data = norm(data)
        return data

    def interpolate_undo_normalize(self, data):
        if hasattr(self, 'undo_norm_list') == False or len(self.undo_norm_list) <= 0:
            return data 
        
        if isinstance(data, list):
            for d in data:
                for unnorm in self.undo_norm_list:
                    data = unnorm(data)
        else:
            for unnorm in self.undo_norm_list:
                data = unnorm(data)
        return data
    

    def nparray_to_image(self, x, save_path):
        height, width, channel = self.data_shape

        figure = np.zeros( (height * len(x), width * channel, 1) )

        for i in range(len(x)):
            for j in range(channel):
                figure[i*height : (i+1)*height, j*width : (j+1)*width] = x[i][..., j : (j+1)]
        
        figure = np.squeeze(figure)

        figure = np.clip(figure * 255, 0, 255).astype('uint8')      # 将0-1变为0-255，似乎很多软件只能打开0-255式的
        img = Image.fromarray(figure)

        postfix = os.path.splitext(save_path)[1]
        if postfix.upper() == '.JPG' or postfix.upper() == '.PNG':
            img = img.convert("RGB")
        # img.show()
        img.save(save_path)
    

    def compute_step_per_epoch(self, tdp, vdp):
        train_data_nums = name_visibility_date_dir_data_counts(tdp, black_list=[self.validate_seperation, 'Invisible'])
        validate_data_nums = name_visibility_date_dir_data_counts(tdp, black_list=['Invisible']) - train_data_nums

        steps_per_epoch = math.floor( train_data_nums / (self.batch_size - 1 + self.block_nums) ) * (len(self.dam) + 1)
        steps_for_validate = math.floor( validate_data_nums / (self.batch_size - 1 + self.block_nums) )
        return steps_per_epoch, steps_for_validate



if __name__ == "__main__":
    glow = Glow()
    glow.load_weights(os.path.join(Option.glow_model_path, Option.glow_model_name))
    glow.create_norm_list(data_root_path=Option.data_root_path, max_min_path=Option.max_min_norm_path)

    hurricane_interpolate = HurricaneInterpolation(batch_size=4, 
                                                block_nums=2,
                                                glow_model=glow.encoder, 
                                                glow_inverse_model=glow.decoder)
    hurricane_interpolate.train(epochs=2, 
                                sample_interval=1, 
                                tdp=Option.data_root_path, 
                                vdp=Option.data_root_path, 
                                smp=Option.interpolate_model_path, 
                                itp=Option.interpolate_result_path)
    # tmp = hurricane_interpolate.build_model()
    # tmp.summary(line_length=150)