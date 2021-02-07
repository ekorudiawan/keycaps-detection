import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from skimage.feature import hog
from scipy.spatial import distance

def main():
    # Image size
    img_width = 100
    img_height = 100
    # HOG orientation
    n_orientations = 9
    # HOG cells
    x_cells = 10
    y_cells = 10
    # ROI Param
    trim_in_top_left = (30,565)
    trim_in_bottom_right = (80,620)
    trim_out_top_left = (140,565)
    trim_out_bottom_right = (180,620)
    red_color = (0,0,255)
    green_color = (0,255,0)
    font = cv.FONT_HERSHEY_SIMPLEX 
    # Loading template ROI
    temp_trim_in = cv.resize(cv.imread("../templates/temp1.png", 0), (100,100))
    temp_trim_out = cv.resize(cv.imread("../templates/temp2.png", 0), (100,100))
    # HOG Feature Detector
    hog_trim_in, hog_trim_in_img = hog(temp_trim_in, orientations = n_orientations,
                                       pixels_per_cell = (img_width//x_cells, img_height//y_cells),
                                       cells_per_block = (x_cells, y_cells), visualize = True)
    hog_trim_out, hog_trim_out_img = hog(temp_trim_out, orientations = n_orientations,
                                         pixels_per_cell = (img_width//x_cells, img_height//y_cells),
                                         cells_per_block = (x_cells, y_cells), visualize = True)
    done = False
    img_number = 0
    while not done:
        test_img = cv.imread("../images/img"+str(img_number)+".png")

        # GET ROI
        test_trim_in = test_img[trim_in_top_left[1]:trim_in_bottom_right[1], trim_in_top_left[0]:trim_in_bottom_right[0]].copy()
        test_trim_out = test_img[trim_out_top_left[1]:trim_out_bottom_right[1], trim_out_top_left[0]:trim_out_bottom_right[0]].copy()
        test_trim_in = cv.resize(test_trim_in, (100,100))
        test_trim_out = cv.resize(test_trim_out, (100,100))

        # Get HOG
        test_hog_trim_in, _ = hog(test_trim_in, orientations = n_orientations,
                                  pixels_per_cell = (img_width//x_cells, img_height//y_cells),
                                  cells_per_block = (x_cells, y_cells), visualize = True)
        test_hog_trim_out, _ = hog(test_trim_out, orientations = n_orientations,
                                   pixels_per_cell = (img_width//x_cells, img_height//y_cells),
                                   cells_per_block = (x_cells, y_cells), visualize = True)

        # Check similarity
        dist_trim_in = distance.euclidean(test_hog_trim_in, hog_trim_in)
        print("Trim In Similarity:", dist_trim_in)
        dist_trim_out = distance.euclidean(test_hog_trim_out, hog_trim_out)
        print("Trim Out Similarity:", dist_trim_out)
        
        cv.putText(test_img,  "{:.2f}".format(dist_trim_in), trim_in_top_left, font, 0.8, (0,0,255), 1, cv.LINE_AA)
        cv.putText(test_img,  "{:.2f}".format(dist_trim_out), trim_out_top_left, font, 0.8, (0,0,255), 1, cv.LINE_AA)

        # Pass
        if dist_trim_in < 0.5:
            cv.rectangle(test_img, trim_in_top_left, trim_in_bottom_right, green_color, thickness=1)
        # Fail
        else:
            cv.rectangle(test_img, trim_in_top_left, trim_in_bottom_right, red_color, thickness=1)
        
        if dist_trim_out < 0.5:
            cv.rectangle(test_img, trim_out_top_left, trim_out_bottom_right, green_color, thickness=1)
        else:
            cv.rectangle(test_img, trim_out_top_left, trim_out_bottom_right, red_color, thickness=1)

        cv.imshow('Testing Result',test_img)
        # cv.imshow('roi',test_trim_in)
        k = cv.waitKey(0)
        # Tekan x untuk exit
        if k == ord('x'):         
            done = True
            cv.destroyAllWindows()
        # Tekan n untuk next image
        elif k == ord('n'):
            img_number += 1
            if img_number > 1:
                img_number = 0
        # Tekan p untuk previous image
        elif k == ord('p'):
            img_number -= 1
            if img_number < 0:
                img_number = 1
        elif k == ord('s'):
            cv.imwrite("../templates/temp1.png", test_trim_in)
            cv.imwrite("../templates/temp2.png", test_trim_out)

if __name__ == "__main__":
    main()