import numpy as np
import cv2
from keras.utils import Sequence
import math
from keras.callbacks import ModelCheckpoint
from keras import optimizers
from keras.models import Model
from keras.layers import Input, Lambda
from keras.optimizers import Adam
import u_net
from keras.models import load_model


class SequenceData(Sequence):

    def __init__(self, data_x, data_y, batch_size):
        self.batch_size = batch_size
        self.data_x = data_x
        self.data_y = data_y
        self.indexes = np.arange(len(self.data_x))

    def __len__(self):
        return math.floor(len(self.data_x) / float(self.batch_size))

    def on_epoch_end(self):
        np.random.shuffle(self.indexes)

    def __getitem__(self, idx):

        batch_index = self.indexes[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_x = [self.data_x[k] for k in batch_index]
        batch_y = [self.data_y[k] for k in batch_index]

        u_size = [192, 192, 3]
        x = np.zeros((self.batch_size, u_size[0], u_size[1], 3))
        y = np.zeros((self.batch_size, u_size[0], u_size[1], 2))

        for i in range(self.batch_size):

            img1 = cv2.imread(batch_x[i])
            size = img1.shape
            resize_img = cv2.resize(img1, (u_size[1], u_size[0]), interpolation=cv2.INTER_AREA)

            x[i, :, :, :] = resize_img/255

            x1 = int(int(batch_y[i][0]) / size[1] * u_size[1])
            y1 = int(int(batch_y[i][1]) / size[0] * u_size[0])
            x2 = int(int(batch_y[i][2]) / size[1] * u_size[1])
            y2 = int(int(batch_y[i][3]) / size[0] * u_size[0])

            for j in range(u_size[0]):
                for k in range(u_size[1]):

                    if y1 <= j <= y2 and x1 <= k <= x2:
                        y[i, j, k, 1] = 1
                    else:
                        y[i, j, k, 0] = 1

        return x, y


# create model and train and save
def train_network(train_generator, validation_generator, epoch):

    model = u_net.create_network()
    model.summary()
    model.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])

    checkpoint = ModelCheckpoint('best_weights.hdf5', monitor='val_loss',
                                 save_weights_only=True, save_best_only=True)

    model.fit_generator(
        train_generator,
        steps_per_epoch=len(train_generator),
        epochs=epoch,
        validation_data=validation_generator,
        validation_steps=len(validation_generator),
        callbacks=[checkpoint]
    )

    model.save_weights('first_weights.hdf5')


def load_network_then_train(train_generator, validation_generator, epoch, input_name, output_name):

    model = u_net.create_network()
    model.load_weights('best_weights.hdf5')
    model.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])

    checkpoint = ModelCheckpoint('best_weights.hdf5', monitor='val_loss',
                                 save_weights_only=True, save_best_only=True)

    model.fit_generator(
        train_generator,
        steps_per_epoch=len(train_generator),
        epochs=epoch,
        validation_data=validation_generator,
        validation_steps=len(validation_generator),
        callbacks=[checkpoint]
    )

    model.save_weights(output_name)


