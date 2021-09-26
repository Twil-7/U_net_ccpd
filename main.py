import numpy as np
import cv2
import read_data_path as rp
import train as tr
from keras.models import load_model
import u_net
from keras.models import load_model
from keras.models import Model
import detect


if __name__ == "__main__":

    train_x, train_y, test_x, test_y = rp.make_data()

    # for i in range(len(test_x)):
    #
    #     img1 = cv2.imread(test_x[i])
    #     size = img1.shape
    #     img = cv2.resize(img1, (84, 84), interpolation=cv2.INTER_AREA)
    #
    #     x1 = int(int(test_y[i][0])/size[1]*84)
    #     y1 = int(int(test_y[i][1])/size[0]*84)
    #     x2 = int(int(test_y[i][2])/size[1]*84)
    #     y2 = int(int(test_y[i][3])/size[0]*84)
    #
    #     cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
    #
    #     cv2.namedWindow("Image")
    #     cv2.imshow("Image", img)
    #     cv2.waitKey(0)

    train_generator = tr.SequenceData(train_x, train_y, 32)
    test_generator = tr.SequenceData(test_x, test_y, 32)

    # tr.train_network(train_generator, test_generator, epoch=10)
    # tr.load_network_then_train(train_generator, test_generator, epoch=10,
    #                            input_name='best_weights.hdf5', output_name='second_weights.hdf5')

    # detect.detect_test_data(test_x)
    # detect.detect_video()
    # detect.detect_usb_video()




