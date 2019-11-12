import cv2
import numpy as np
import os
from PIL import Image
from skimage import feature, exposure
from sklearn import svm


def train_red():
    folder_train_red = [["train_set/00001", 1], ["train_set/00014", 14], ["train_set/00017", 17],
                        ["train_set/00019", 19], ["train_set/00021", 21], ["negative", 100]]

    hog_list_red = list()
    label_list_red = list()
    count_red = 0

    for name_red in folder_train_red:
        value_red = name_red[0]
        label_red = name_red[1]
        image_list_red = [os.path.join(value_red, f) for f in os.listdir(value_red) if f.endswith('.ppm')]

        for image_red in image_list_red:
            count_red += 1
            # print(count)
            im_red = np.array(Image.open(image_red))
            im_gray_red = cv2.cvtColor(im_red, cv2.COLOR_BGR2GRAY)
            im_prep_red = cv2.resize(im_gray_red, (64, 64))

            fd_red, h_red = feature.hog(im_prep_red, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(4, 4),
                                        transform_sqrt=False, block_norm="L1", visualise=True)
            hog_list_red.append(h_red)
            label_list_red.append(label_red)

    list_hogs_red = []
    for hogs_red in hog_list_red:
        hogs_red = hogs_red.reshape(64 * 64)
        list_hogs_red.append(hogs_red)

    clf_red = svm.SVC(gamma='scale', probability=True, decision_function_shape='ovo')
    clf_red.fit(list_hogs_red, label_list_red)

    return clf_red


def test_red(clf_red, image):
    im_test_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    fd_test_red, h_test_red = feature.hog(im_test_gray, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(4, 4),
                                          transform_sqrt=False, block_norm="L1", visualise=True)

    hog = h_test_red.reshape(64 * 64)
    predict = clf_red.predict([hog])
    class_prob = clf_red.predict_proba([hog])

    return predict, class_prob


def train_blue():
    folder_train_blue = [["train_set/00035", 35], ["train_set/00038", 38], ["train_set/00045", 45], ["negative", 100]]

    hog_list_blue = list()
    label_list_blue = list()
    count_blue = 0

    for name_blue in folder_train_blue:
        value_blue = name_blue[0]
        label_blue = name_blue[1]
        image_list_red = [os.path.join(value_blue, f) for f in os.listdir(value_blue) if f.endswith('.ppm')]

        for image_blue in image_list_red:
            count_blue += 1
            # print(count)
            im_blue = np.array(Image.open(image_blue))
            im_gray_blue = cv2.cvtColor(im_blue, cv2.COLOR_BGR2GRAY)
            im_prep_blue = cv2.resize(im_gray_blue, (64, 64))

            fd_blue, h_blue = feature.hog(im_prep_blue, orientations=7, pixels_per_cell=(8, 8), cells_per_block=(2, 2),
                                          transform_sqrt=False, block_norm="L1", visualise=True)
            cv2.imshow('hog_train', h_blue)
            cv2.waitKey(50)
            hog_list_blue.append(h_blue)
            label_list_blue.append(label_blue)

    list_hogs_blue = []
    for hogs_blue in hog_list_blue:
        hogs_blue = hogs_blue.reshape(64 * 64)
        list_hogs_blue.append(hogs_blue)

    clf_blue = svm.SVC(gamma='scale', probability=True, decision_function_shape='ovo')
    clf_blue.fit(list_hogs_blue, label_list_blue)

    return clf_blue


def test_blue(clf_blue, image):
    im_test_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    fd_test_blue, h_test_blue = feature.hog(im_test_gray, orientations=7, pixels_per_cell=(8,8),
                                            cells_per_block=(2, 2), transform_sqrt=False, block_norm="L1", visualise=True)

    hog = h_test_blue.reshape(64 * 64)
    predict = clf_blue.predict([hog])
    class_prob = clf_blue.predict_proba([hog])

    return predict, class_prob


# def train_negative():
#     folder_train_negative = [["negative", 100]]

#     hog_list_negative = list()
#     label_list_negative = list()
#     count_red = 0

#     for name_negative in folder_train_negative:
#         value_negative = name_negative[0]
#         label_negative = name_negative[1]
#         image_list_negative = [os.path.join(value_negative, f) for f in os.listdir(value_negative) if f.endswith('.jpg')]

#         for image_negative in image_list_negative:
#             count_negative += 1
#             # print(count)
#             im_negative = np.array(Image.open(image_negative))
#             im_gray_negative = cv2.cvtColor(im_negative, cv2.COLOR_BGR2GRAY)
#             im_prep_negative = cv2.resize(im_gray_negative, (64, 64))

#             fd_negative, h_negative = feature.hog(im_prep_negative, orientations=7, pixels_per_cell=(8, 8), cells_per_block=(2, 2),
#                                         transform_sqrt=False, block_norm="L1", visualise=True)
#             hog_list_negative.append(h_negative)
#             label_list_negative.append(label_negative)

#     list_hogs_negative = []
#     for hogs_negative in hog_list_negative:
#         hogs_negative = hogs_negative.reshape(64 * 64)
#         list_hogs_negative.append(hogs_negative)

#     clf_negative = svm.SVC(gamma='scale', probability=True, decision_function_shape='ovo')
#     clf_negative.fit(list_hogs_negative, label_list_negative)

#     return clf_negative
"""clf_blue = train_blue()
folder_test_red = [["test_set/00001", 1], ["test_set/00014", 14], ["test_set/00017", 17], ["test_set/00019", 19],
                   ["test_set/00021", 21]]

folder_test_blue = [["test_set/00035", 35], ["test_set/00038", 38], ["test_set/00045", 45]]


hog_list_test = list()
label_list_test = list()
count_img = 0
for name in folder_test_blue:
    value = name[0]
    label = name[1]
    image_list = [os.path.join(value, f) for f in os.listdir(value) if f.endswith('.ppm')]

    for image in image_list:
        count_img += 1
        im = np.array(Image.open(image))
        im_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        im_prep = cv2.resize(im_gray, (64, 64))
        #cv2.imshow('first', np.uint8(im_prep))

        fd, h = feature.hog(im_prep, orientations=7, pixels_per_cell=(8, 8), cells_per_block=(2, 2),
                            transform_sqrt=False, block_norm="L1", visualise=True)

        cv2.imshow('hog_test', h)
        cv2.waitKey(50)

        hog_list_test.append(h)
        label_list_test.append(label)

list_hogs_test = []
for hogs in hog_list_test:
    hogs = hogs.reshape(64*64)
    list_hogs_test.append(hogs)

predictions = clf_blue.predict(list_hogs_test)
class_prob = clg_blue.predict_proba()

#print(predictions)

count = 0
for i in range(len(predictions)):
    if predictions[i] == label_list_test[i]:
        count += 1

# total number of test images = 884
print("Percentage accuracy: ", (count/count_img)*100)"""











