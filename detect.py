import numpy as np
import cv2
import u_net


def detect_img(model, img, u_size):

    img1 = img
    size = img1.shape
    img2 = cv2.resize(img1, (u_size[1], u_size[0]), interpolation=cv2.INTER_AREA)
    img3 = img2 / 255
    img4 = img3[np.newaxis, :, :, :]

    result = model.predict(img4)

    # cv2.namedWindow("img3")
    # cv2.imshow("img3", img3)
    # cv2.waitKey(0)
    #
    # cv2.namedWindow("mask")
    # cv2.imshow("mask", mask)
    # cv2.waitKey(0)

    x1 = u_size[1] - 1
    y1 = u_size[1] - 1
    x2 = 0
    y2 = 0

    for j in range(u_size[0]):
        for k in range(u_size[1]):

            index = np.argmax(result[0, j, k, :])

            if index == 1:

                if x1 > k:
                    x1 = k
                if y1 > j:
                    y1 = j
                if x2 < k:
                    x2 = k
                if y2 < j:
                    y2 = j

    print(x1, y1, x2, y2)

    a1 = int(x1 / u_size[1] * size[1])
    b1 = int(y1 / u_size[0] * size[0])
    a2 = int(x2 / u_size[1] * size[1])
    b2 = int(y2 / u_size[0] * size[0])

    cv2.rectangle(img1, (a1, b1), (a2, b2), (0, 0, 255), 2)

    # cv2.namedWindow("Image")
    # cv2.imshow("Image", img1)
    # cv2.waitKey(0)

    return img1


def detect_test_data(test_x):

    model = u_net.create_network()
    model.load_weights('best_weights.hdf5')
    u_size = [192, 192, 3]

    for i in range(len(test_x)):

        img = cv2.imread(test_x[i])
        final_img = detect_img(model, img, u_size)
        cv2.imwrite("test_demo/" + str(i) + '.jpg', final_img)


def detect_video():

    cap = cv2.VideoCapture('car.mp4')
    video_num = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    four = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter('detect_video.mp4', four, 20.0, (400, 420))

    model = u_net.create_network()
    model.load_weights('best_weights.hdf5')
    u_size = [192, 192, 3]

    frame_count = 0

    while frame_count < video_num:

        success, img = cap.read()             # (960, 544, 3)

        crop_img = img[240:660, 44:444, :]    # (420, 400, 3)
        final_img = detect_img(model, crop_img, u_size)

        cv2.imwrite("video_demo/" + str(frame_count) + '.jpg', final_img)
        out.write(final_img)
        frame_count = frame_count + 1

    cap.release()


def detect_usb_video():

    model = u_net.create_network()
    model.load_weights('best_weights.hdf5')
    u_size = [192, 192, 3]

    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        print(ret)

        img = frame[:, :, :]     # (420, 400, 3)
        final_img = detect_img(model, img, u_size)

        cv2.imshow('Live Video', final_img)
        cv2.waitKey(1)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

