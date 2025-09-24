import tensorflow as tf
import cv2
import os
import numpy as np
import csv

label_color = {"head": [255, 0, 255], "shadow": [0, 255, 0]}
Image_Path = ".//360TRUE//"
Mask_Path = ".//PredictMask_360//"
Save_Analyse_Path = ".//AnalyseShape_360//"
cvs_path = ".//AnalyseInformation_360.csv"
error_path = ".//ErrorFile360.csv"
ana_size = (128, 128)
method_all_max = "all"  # "all" or ”max“


def image_resize(img, size):
    ih, iw = img.shape[:2]
    w, h = size

    scale = min(w / iw, h / ih)
    nw = int(iw * scale)
    nh = int(ih * scale)
    dst = cv2.resize(img, (nw, nh))
    board = np.zeros((h, w, 3), dtype=np.uint8)

    w_start = (w - nw) // 2
    h_start = (h - nh) // 2
    board[h_start:h_start + nh, w_start:w_start + nw] = dst
    ratio = 1/scale 
    return board, ratio


def image_resize_main(img, size):
    ih, iw = img.shape[:2]
    w, h = size

    scale = min(2 * w / iw, 2 * h / ih)
    nw = int(iw * scale)
    nh = int(ih * scale)
    dst = cv2.resize(img, (nw, nh))
    board = np.zeros((3*h, 3*w, 3), dtype=np.uint8)

    w_start = (3*w - nw) // 2
    h_start = (3*h - nh) // 2
    board[h_start:h_start + nh, w_start:w_start + nw] = dst
    ratio = 1/scale 
    cs = [ratio, h_start, w_start]
    return board, cs


def make_small_show(img, mask, size):
    img, ra = image_resize(img, size)
    mask, ra = image_resize(mask, size)

    combine = cv2.addWeighted(img, 0.5, mask, 0.5, 0)
    imshow_img = np.concatenate([img, mask, combine], axis=0)
    return imshow_img


def make_lager_show(img, mask, size):
    img_ana, ra = image_resize_main(img, size)
    mask_ana, ra = image_resize_main(mask, size)
    return img_ana, mask_ana, ra


csvF = open(cvs_path,'w',newline="")
csv_writer = csv.writer(csvF)
csv_writer.writerow(["NAME", "Type", "write_x", "write_y", "A", "B", "Angle"])

csvF_ERROR = open(error_path,'w',newline="")
csv_ERROR_writer = csv.writer(csvF_ERROR)

for i, j, k in os.walk(Mask_Path): 
    catalogList = k  

for img_name in catalogList:
    try:
        img_org = cv2.imread(Image_Path + img_name)
        mask_org = cv2.imread(Mask_Path + img_name)

        imshow_img_left = make_small_show(img_org, mask_org, ana_size)
        img_ana, mask_ana, cs = make_lager_show(img_org, mask_org, ana_size)

        draw_fill_board = np.zeros_like(img_ana)
        fitting_list = {}
        for ind, type_object in enumerate(label_color.keys()):
            ana_mask_obj = mask_ana == label_color[type_object]
            ana_mask_obj = ana_mask_obj[:, :, 0] * ana_mask_obj[:, :, 1] * ana_mask_obj[:, :, 2]
            ana_mask_obj = ana_mask_obj.astype(np.uint8) * 255
            contours, hierarchy = cv2.findContours(ana_mask_obj, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

            if method_all_max == "all":
                cnt = np.concatenate(contours, axis=0)
            else:
                iii = [len(ssss) for ssss in contours]
                cnt = contours[np.argmax(iii)]

            cv2.drawContours(draw_fill_board, [cnt], -1, label_color[type_object], -1)
            ellipse_fit = cv2.fitEllipse(cnt)
            fitting_list[type_object] = ellipse_fit

        img_ana = cv2.addWeighted(img_ana, 0.7, draw_fill_board, 0.3, 0)
        for ind, type_object in enumerate(label_color.keys()):
            ellipse_fit = fitting_list[type_object]

            center_x = ellipse_fit[0][0]
            center_y = ellipse_fit[0][1]
            a = ellipse_fit[1][0]
            b = ellipse_fit[1][1]
            angle = ellipse_fit[2]#如果 angle 是负值，则表示椭圆是水平放置的，但是方向是向左的。如果 angle 是正值，则表示椭圆是水平放置的，方向是向右的。
            if b > a:
                a, b = b, a
                angle = angle - 90
            cv2.ellipse(img_ana, ((center_x, center_y), (a, b), angle), label_color[type_object], 1)
            ratio, img_start_h, img_start_w = cs
            write_x = (center_x - cs[2]) * cs[0]
            write_y = (center_y - cs[1]) * cs[0]
            write_a = a * ratio
            write_b = b * ratio

            cv2.putText(img_ana, type_object, (ind*int(3*ana_size[0]/2), 10), cv2.FONT_HERSHEY_COMPLEX, 0.5, label_color[type_object], 1)
            cv2.putText(img_ana, "x:%.2f"%write_x, (ind * int(3 * ana_size[0] / 2), 22), cv2.FONT_HERSHEY_COMPLEX, 0.5,label_color[type_object], 1)
            cv2.putText(img_ana, "y:%.2f" % write_y, (ind * int(3 * ana_size[0] / 2), 34), cv2.FONT_HERSHEY_COMPLEX, 0.5, label_color[type_object], 1)
            cv2.putText(img_ana, "a:%.2f" % write_a, (ind * int(3 * ana_size[0] / 2), 46), cv2.FONT_HERSHEY_COMPLEX, 0.5, label_color[type_object], 1)
            cv2.putText(img_ana, "b:%.2f" % write_b, (ind * int(3 * ana_size[0] / 2), 58), cv2.FONT_HERSHEY_COMPLEX, 0.5, label_color[type_object], 1)
            cv2.putText(img_ana, "angle:%.2f" % angle, (ind * int(3 * ana_size[0] / 2), 70), cv2.FONT_HERSHEY_COMPLEX, 0.5, label_color[type_object], 1)
            csv_writer.writerow([img_name, type_object, write_x, write_y, write_a, write_b, angle])

        imshow_img = np.concatenate([imshow_img_left, img_ana], axis=1)
        cv2.imwrite(Save_Analyse_Path + img_name, imshow_img)
        print('\033[32m' + "Success: %s"%img_name + '\033[0m')
    except:
        print('\033[31m' + "Error: %s"%img_name  + '\033[0m')
        csv_ERROR_writer.writerow([img_name])

csvF.close()
csvF_ERROR.close()


