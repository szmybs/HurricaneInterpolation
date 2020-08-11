import numpy as np
import os
import math
import time

from PIL import Image

from keras import backend as K
from keras.layers import *
from keras.models import Model
from keras.callbacks import Callback, ModelCheckpoint
from keras.optimizers import Adam

from flow_layers import Squeeze

from hurricane_generator import name_visibility_date_dir_generator, name_visibility_date_dir_data_counts
from extract import HurricaneExtraction


class Glow(object):
    def __init__(self):
        self.batch_size = 8
        self.learning_rate = 1e-4

        self.level = 3
        self.depth = 10

        self.validate_seperation = 'LIDIA'

        self.data_shape = (256, 256, 5)


    def build_basic_model(self, in_channel):
        """基础模型，即耦合层中的模型（basic model for Coupling）
        """
        _in = Input(shape=(None, None, in_channel))
        _ = _in
        hidden_dim = 512
        _ = Conv2D(hidden_dim,
                (3, 3),
                padding='same')(_)
        # _ = Actnorm(add_logdet_to_loss=False)(_)
        _ = Activation('relu')(_)
        _ = Conv2D(hidden_dim,
                (1, 1),
                padding='same')(_)
        # _ = Actnorm(add_logdet_to_loss=False)(_)
        _ = Activation('relu')(_)
        _ = Conv2D(in_channel,
                (3, 3),
                kernel_initializer='zeros',
                padding='same')(_)
        return Model(_in, _)
    

    def compile(self):
        optimizer = Adam(lr=self.learning_rate)
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
                couple = CoupleWrapper(build_basic_model(3*2**(i+1)))  #?????
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

        self.encoder = Model(x_in, x)
        for l in self.encoder.layers:
            if hasattr(l, 'logdet'):
                self.encoder.add_loss(l.logdet)

        self.encoder.summary()


        # decoder part
        x_in = Input(shape=K.int_shape(self.encoder.outputs[0])[1:])
        x = x_in

        x = final_concat.inverse()(x)
        outputs = x[:-1]
        x = x[-1]
        x = final_reshape.inverse()(x)
        x = final_actnorm.inverse()(x)
        x1 = x

        for i,(split,condactnorm,reshape) in enumerate(zip(*outer_layers)[::-1]):
            if i > 0:
                x1 = x
                x_out = outputs[-i]
                x_out = reshape.inverse()(x_out)
                x2 = condactnorm.inverse()([x_out, x1])
                x = split.inverse()([x1, x2])
            for j,(actnorm,permute,split,couple,concat) in enumerate(zip(*inner_layers)[::-1][i*self.depth: (i+1)*self.depth]):
                x1, x2 = concat.inverse()(x)
                x1, x2 = couple.inverse()([x1, x2])
                x = split.inverse()([x1, x2])
                x = permute.inverse()(x)
                x = actnorm.inverse()(x)
            x = squeeze.inverse()(x)

        self.decoder = Model(x_in, x)

        self.decoder.summary()


    def nparray_to_image(self, x):
        height, width, channel = self.data_shape
        # x = HurricaneExtraction.convert_unsigned_to_float(x)
        # x = HurricaneExtraction.normalize_using_physics(x)

        figure = np.zeros( (height * len(x), width * channel, 1) )

        for i in range(len(x)):
            for j in range(channel):
                figure[i*height : (i+1)*height, j*width : (j+1)*width] = x[i][..., j : (j+1)]
        
        figure = np.squeeze(figure)
        img = Image.fromarray(figure)
        return img


    # def sample(self, path, std=1):
    #     height, width, channel = self.data_shape[0], self.data_shape[1], self.data_shape[2]
        
    #     for i in range()


    # def sample(self, path, img_size, n=9, std=1):
    #     image_size = self.data_shape

    #     figure = np.zeros((height * n, width * n, channel))
    #     for i in range(n):
    #         for j in range(n):
    #             decoder_input_shape = (1,) + K.int_shape(self.decoder.inputs[0])[1:]
    #             z_sample = np.array(np.random.randn(*decoder_input_shape)) * std
    #             x_decoded = self.decoder.predict(z_sample)
    #             digit = x_decoded[0].reshape(img_size)

    #             # 这里height width顺序可能有误
    #             figure[i * height: (i + 1) * height, j * width: (j + 1) * width] = digit

    #     figure = (figure + 1) / 2 * 255
    #     figure = np.clip(figure, 0, 255).astype('uint8')
    #     imageio.imwrite(path, figure)


    def reconsturt(self, data_root_path, save_path, epoch):
        x = []

        validate_data_generator = self.glow_generator(data_root_path, mode='validate')
        for i in range(16):
            vd = next(validate_data_generator)
            lv = self.encoder.predict(vd)
            re = self.decoder.predict(lv)
            x.append[re]
        img = self.nparray_to_image(re)
        save_name = str(epoch) + ".tiff"
        img.save(os.path.join(save_path, save_name))


    def glow_generator(self, data_root_path, mode = 'train'):
        if mode == 'train':
            gg = name_visibility_date_dir_generator(data_root_path, hurricane_name_black_list=self.validate_seperation)
        elif mode == 'validate':
            gg = name_visibility_date_dir_generator(data_root_path, hurricane_name_white_list=self.validate_seperation)
        else:
            gg = name_visibility_date_dir_generator(data_root_path)

        while True:
            gd = next(gg)
            gd = np.reshape( (gd.shape[0], -1) )
            yield(gd)


    def train(self, epochs, sample_interval=10, **kwarg):
        data_root_path = kwarg['drp']
        save_model_path = kwarg['swp']

        if kwarg['srp'] is not None:
            sample_root_path = kwarg['srp']

        # model
        self.build_model(self.data_shape)
        self.compile()

        # generator
        train_data_generator = self.glow_generator(data_root_path)
        validate_data_generator = self.glow_generator(data_root_path, mode='validate')

        # epoch
        train_data_nums = name_visibility_date_dir_data_counts(data_root_path, black_list=[self.validate_seperation])
        validate_data_nums = name_visibility_date_dir_data_counts(data_root_path) - train_data_nums

        steps_per_epoch = math.floor( train_data_nums / self.batch_size )
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
        class Evaluate(Callback):
            def __init__(self, save_model_path, sample_root_path, sample_func):
                self.lowest = 1e10
                self.save_model_path = save_model_path
                self.sample_root_path = sample_root_path
                self.sample_func = sample_func

            def on_epoch_end(self, epoch, logs=None):
                if logs['loss'] <= self.lowest:
                    self.lowest = logs['loss']
                    self.model.save_weights(os.path.join(self.save_model_path,'best_encoder.weights'))
                elif logs['loss'] > 0 and epoch > 10:
                    """在后面，loss一般为负数，一旦重新变成正数，就意味着模型已经崩溃，需要降低学习率。"""
                    self.model.load_weights(os.path.join(self.save_model_path,'best_encoder.weights'))
                    K.set_value(self.model.optimizer.lr, self.model.optimizer.lr * 0.1)

        evaluate = Evaluate(save_model_path, self.sample, self.sample)

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
