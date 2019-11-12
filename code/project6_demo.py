import numpy as np
import os
import cv2
import imutils
import classification as train
from PIL import Image

# Calculate the Classifier from train path
clf_red = train.train_red()  # model for red
clf_blue = train.train_blue()  # model for blue
#clf_negative = train.train_negative() # model for negative
# to create the dict for pasting the image in the final result
folder_paste = [["train_set/00001", 1], ["train_set/00014", 14], ["train_set/00017", 17], ["train_set/00019", 19],
                ["train_set/00021", 21], ["train_set/00035", 35], ["train_set/00038", 38], ["train_set/00045", 45]]

img_list = {}
for name in folder_paste:
    label_paste = name[1]
    value_paste = name[0]
    image_list = [os.path.join(value_paste, f) for f in os.listdir(value_paste) if f.endswith('.ppm')]
    img = np.array(Image.open(image_list[0]))
    img_list[label_paste] = cv2.resize(img, (64, 64))


def identify_blue(imag, clf_blue):
    label_list = list()
    cnts_list = list()
    mser_blue = cv2.MSER_create(8, 400, 4000)

    img = imag.copy()
    img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)

    # equalize the histogram of the Y channel
    img_yuv[:, :, 0] = cv2.equalizeHist(img_yuv[:, :, 0])

    # convert the YUV image back to RGB format
    img_output = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)

    # convert the image to HSV format for color segmentation
    img_hsv = cv2.cvtColor(imag, cv2.COLOR_BGR2HSV)

    # mask to extract blue
    lower_blue = np.array([94, 127, 20])
    upper_blue = np.array([126, 255, 200])
    mask = cv2.inRange(img_hsv, lower_blue, upper_blue)

    blue_mask = cv2.bitwise_and(img_output, img_output, mask=mask)

    # seperate out the channels
    r_channel = blue_mask[:, :, 2]
    g_channel = blue_mask[:, :, 1]
    b_channel = blue_mask[:, :, 0]

    # filter out
    filtered_r = cv2.medianBlur(r_channel, 5)
    filtered_g = cv2.medianBlur(g_channel, 5)
    filtered_b = cv2.medianBlur(b_channel, 5)

    # create a blue gray space
    filtered_b = -0.5 * filtered_r + 3 * filtered_b - 2 * filtered_g

    # Do MSER
    regions, _ = mser_blue.detectRegions(np.uint8(filtered_b))

    hulls = [cv2.convexHull(p.reshape(-1, 1, 2)) for p in regions]

    blank = np.zeros_like(blue_mask)
    cv2.fillPoly(np.uint8(blank), hulls, (255, 0, 0))

    # cv2.imshow("mser_blue", blank)
    kernel_1 = np.ones((3, 3), np.uint8)
    kernel_2 = np.ones((5, 5), np.uint8)

    erosion = cv2.erode(blank, kernel_1, iterations=1)
    dilation = cv2.dilate(erosion, kernel_2, iterations=1)
    opening = cv2.morphologyEx(dilation, cv2.MORPH_OPEN, kernel_2)

    _, b_thresh = cv2.threshold(opening[:, :, 0], 60, 255, cv2.THRESH_BINARY)

    cnts = cv2.findContours(b_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    max_cnts = 3  # no frame we want to detect more than 3

    if not cnts == []:
        cnts_sorted = sorted(cnts, key=cv2.contourArea, reverse=True)
        if len(cnts_sorted) > max_cnts:
            cnts_sorted = cnts_sorted[:3]

        for c in cnts_sorted:
            x, y, w, h = cv2.boundingRect(c)
            if x < 100:
                continue
            if h < 20:
                continue

            if y > 400:
                continue

            aspect_ratio_1 = w / h
            aspect_ratio_2 = h / w
            if aspect_ratio_1 <= 0.5 or aspect_ratio_1 > 1.2:
                continue
            if aspect_ratio_2 <= 0.5:
                continue

            hull = cv2.convexHull(c)

            # cv2.rectangle(imag, (x, y), (int(x+w), int(y+h)), (0, 255, 0), 2)
            # cv2.drawContours(imag, [hull], -1, (0, 255, 0), 2)

            mask = np.zeros_like(imag)
            # cv2.drawContours(mask, [c], -1, (255, 255, 255), -1)  # Draw filled contour in mask
            cv2.rectangle(mask, (x, y), (int(x + w), int(y + h)), (255, 255, 255), -1)
            out = np.zeros_like(imag)  # Extract out the object and place into output image
            out[mask == 255] = imag[mask == 255]

            x_pixel, y_pixel, _ = np.where(mask == 255)
            (topx, topy) = (np.min(x_pixel), np.min(y_pixel))
            (botx, boty) = (np.max(x_pixel), np.max(y_pixel))
            if np.abs(topx - botx) <= 25 or np.abs(topy - boty) <= 25:
                continue

            out = imag[topx:botx + 1, topy:boty + 1]
            out_resize = cv2.resize(out, (64, 64), interpolation=cv2.INTER_CUBIC)
            predict, prob = train.test_blue(clf_blue, out_resize)
            print(np.max(prob))
            if np.max(prob) < 0.78:
                continue
            #cv2.rectangle(imag, (x, y), (int(x + w), int(y + h)), (0, 255, 0), 2)
            label = predict[0]
            if label == 100:
                continue
            
            cnts_list.append(c)
            label_list.append(label)
        return cnts_list, label_list
    else:
        return None, None


def identify_red(imag, clf_red):
    label_list = list()
    cnts_list = list()
    mser_red = cv2.MSER_create(8, 200, 3000)

    img = imag.copy()

    img2 = imag.copy()[:500, :]  # red signs are only on the above few rows
    img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)

    # equalize the histogram of the Y channel
    img_yuv[:, :, 0] = cv2.equalizeHist(img_yuv[:, :, 0])

    # convert the YUV image back to RGB format
    img_output = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)

    # mask to extract red
    img_hsv = cv2.cvtColor(imag, cv2.COLOR_BGR2HSV)
    lower_red_1 = np.array([0, 70, 60])
    upper_red_1 = np.array([10, 255, 255])
    mask_1 = cv2.inRange(img_hsv, lower_red_1, upper_red_1)
    lower_red_2 = np.array([170, 70, 60])
    upper_red_2 = np.array([180, 255, 255])
    mask_2 = cv2.inRange(img_hsv, lower_red_2, upper_red_2)
    mask = cv2.bitwise_or(mask_1, mask_2)
    red_mask_ = cv2.bitwise_and(img_output, img_output, mask=mask)
    red_mask = red_mask_[:500, :]

    # separating channels
    r_channel = red_mask[:, :, 2]
    g_channel = red_mask[:, :, 1]
    b_channel = red_mask[:, :, 0]

    # filtering
    filtered_r = cv2.medianBlur(r_channel, 5)
    filtered_g = cv2.medianBlur(g_channel, 5)
    filtered_b = cv2.medianBlur(b_channel, 5)

    filtered_r = 4 * filtered_r - 0.5 * filtered_b - 2 * filtered_g

    # MSER detection
    regions, _ = mser_red.detectRegions(np.uint8(filtered_r))

    hulls = [cv2.convexHull(p.reshape(-1, 1, 2)) for p in regions]

    blank = np.zeros_like(red_mask)
    cv2.fillPoly(np.uint8(blank), hulls, (0, 0, 255))  # fill a blank image with the detected hulls
    # cv2.imshow("mser_red", blank)
    # perform some operations on the detected hulls from MSER
    kernel_1 = np.ones((3, 3), np.uint8)
    kernel_2 = np.ones((5, 5), np.uint8)

    erosion = cv2.erode(blank, kernel_1, iterations=1)
    dilation = cv2.dilate(erosion, kernel_2, iterations=1)
    opening = cv2.morphologyEx(dilation, cv2.MORPH_OPEN, kernel_2)

    _, r_thresh = cv2.threshold(opening[:, :, 2], 20, 255, cv2.THRESH_BINARY)

    cnts = cv2.findContours(r_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    max_cnts = 3  # no frame we want to detect more than 3
    if not cnts == []:
        cnts_sorted = sorted(cnts, key=cv2.contourArea, reverse=True)
        if len(cnts_sorted) > max_cnts:
            cnts_sorted = cnts_sorted[:3]

        for c in cnts_sorted:
            x, y, w, h = cv2.boundingRect(c)
            if x < 800:
                continue
            aspect_ratio_1 = w / h
            aspect_ratio_2 = h / w
            if aspect_ratio_1 <= 0.3 or aspect_ratio_1 > 1.2:
                continue
            if aspect_ratio_2 <= 0.3:
                continue

            hull = cv2.convexHull(c)
            # cv2.drawContours(imag, [hull], -1, (0, 255, 0), 1)

            mask = np.zeros_like(imag)
            # cv2.drawContours(mask, [c], -1, (255, 255, 255), -1)  # Draw filled contour in mask
            cv2.rectangle(mask, (x, y), (int(x + w), int(y + h)), (255, 255, 255), -1)

            out = np.zeros_like(imag)  # Extract out the object and place into output image
            out[mask == 255] = imag[mask == 255]

            x_pixel, y_pixel, _ = np.where(mask == 255)
            (topx, topy) = (np.min(x_pixel), np.min(y_pixel))
            (botx, boty) = (np.max(x_pixel), np.max(y_pixel))
            if np.abs(topx - botx) <= 25 or np.abs(topy - boty) <= 25:
                continue

            out = imag[topx:botx + 1, topy:boty + 1]
            out_resize = cv2.resize(out, (64, 64), interpolation=cv2.INTER_CUBIC)

            predict, prob = train.test_blue(clf_red, out_resize)
            #print(np.max(prob))
            if np.max(prob) < 0.85:
                continue
            #cv2.rectangle(imag, (x, y), (int(x + w), int(y + h)), (0, 255, 0), 2)
            label = predict[0]
            if label == 100:
                continue 
            cnts_list.append(c)
            label_list.append(label)
        return cnts_list, label_list
    else:
        return None, None


# def identify_negative()

value = "./input"
output_array = []
count = 1
image_list = [os.path.join(value, f) for f in os.listdir(value) if f.endswith('.jpg')]
for image in image_list:
    imag = np.uint8(cv2.imread(image))
    orig = imag.copy()
    imag_blue = imag.copy()
    imag_red = imag.copy()
    cnts_blue_list, label_blue_list = identify_blue(imag_blue, clf_blue)
    cnts_red_list, label_red_list = identify_red(imag_red, clf_red)
    # cnts_negative_list, label_negative_list = identify_negative(imag_negative, clf_negative)

    if cnts_blue_list is not None:
        for i in range(len(cnts_blue_list)):
            x, y, w, h = cv2.boundingRect(cnts_blue_list[i])
            cv2.rectangle(imag, (x, y), (int(x + w), int(y + h)), (0, 255, 0), 2)
            new_x = x - 64
            new_y = y + 64
            imag[y:new_y, new_x:x] = img_list[label_blue_list[i]]
            resized_mask = cv2.resize(imag, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_CUBIC)

    if cnts_red_list is not None:
        for i in range(len(cnts_red_list)):
            x, y, w, h = cv2.boundingRect(cnts_red_list[i])
            cv2.rectangle(imag, (x, y), (int(x + w), int(y + h)), (0, 255, 0), 2)
            new_x = x - 64
            new_y = y + 64
            imag[y:new_y, new_x:x] = img_list[label_red_list[i]]
            resized_mask = cv2.resize(imag, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_CUBIC)

    resized = cv2.resize(imag, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_CUBIC)
    cv2.imwrite("output/frame%d.jpg"%count,resized)
    output_array.append(resized)    
    count = count+1
    cv2.imshow('ope', resized)
    cv2.waitKey(2)


height, width, layers = resized.shape
size = (width,height)
out = cv2.VideoWriter('output.avi',cv2.VideoWriter_fourcc(*'DIVX'), 15.0, size)
for i in range(len(output_array)):
        
        # writing to a image array
    out.write(output_array[i])
out.release()