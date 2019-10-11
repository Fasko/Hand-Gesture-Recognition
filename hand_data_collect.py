'''
# Michael Fasko Jr & Jacob Calfee
# Cleveland State University
# Hackron 4k (10/5/19 - 10/6/19) - Hand Gesture Recognition with Machine Learning

 # File is responsible for taking creating training data from webcam input. Program writes the 21 X,Y coordinates with a
    data label by using OpenPose's hand detector.
 # 9 unique gestures were obtained with the following labels:
    0 --> Front Fist
    1 --> Back Fist
    2 --> Front Peace Sign
    3 --> Back Peace Sign
    4 --> Front Palm
    5 --> Back Palm
    6 --> Thumbs Up
    7 --> Thumbs Down
    8 --> Ok Sign

 # NOTE: Pass '--number_people_max 1' as an argument for best efficiency
 # NOTE: Change the data label on line 109 when appending a new data set to the CSV file
'''

import argparse
import csv
import os
import sys

import cv2

# TODO clean this up
dir_path = "C:\\Users\\Fasko\\Documents\\Complex-Human-Activity-Recognition\\PoseEvaluation"
sys.path.append("{}\\python\\openpose\\Release".format(dir_path))
os.environ["PATH"] = os.environ["PATH"] + ";{}\\Release;{}\\bin".format(dir_path, dir_path)

try:
    import pyopenpose as op
except ImportError as e:
    print(e, file=sys.stderr)

# openpose Flags
parser = argparse.ArgumentParser()

parser.add_argument("--hand", default=True, help="Process an image. Read all standard formats (jpg, png, bmp, etc.).")
parser.add_argument("--display", default=0, help="Process an image. Read all standard formats (jpg, png, bmp, etc.).")

args = parser.parse_known_args()

# openpose Custom Params (refer to include/openpose/flags.hpp for more parameters)
params = dict()
params["model_folder"] = "{}\\models".format(dir_path)
params["hand"] = True
params["display"] = 0
# Add others in path?
for i in range(0, len(args[1])):
    curr_item = args[1][i]
    if i != len(args[1]) - 1:
        next_item = args[1][i + 1]
    else:
        next_item = "1"
    if "--" in curr_item and "--" in next_item:
        key = curr_item.replace('-', '')
        if key not in params:  params[key] = "1"
    elif "--" in curr_item and "--" not in next_item:
        key = curr_item.replace('-', '')
        if key not in params: params[key] = next_item

try:
    # Starting OpenPose
    opWrapper = op.WrapperPython()
    opWrapper.configure(params)
    opWrapper.start()

    datum = op.Datum()

    cap = cv2.VideoCapture(0)  # Use webcam
    num_data_points = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Use webcam frames
        datum.cvInputData = frame
        opWrapper.emplaceAndPop([datum])
        op_frame = datum.cvOutputData

        window_name = "Hand Detector Window"
        cv2.imshow(window_name, op_frame)

        x_coord_list = []
        y_coord_list = []

        rightHandKeypoints = datum.handKeypoints[1]
        conf_lvl_sum = 0
        for entries in rightHandKeypoints:
            for x in entries:
                x_coord_list.append(x[0])
                y_coord_list.append(x[1])
                conf_lvl_sum += (x[2])

        # Build Lists of X,Y + label
        conf_lvl_avg = (conf_lvl_sum/21)

        # Accept the data above conf_level -->  write, append label
        if conf_lvl_avg >= 0.15:
            x_y_coord_list = x_coord_list + y_coord_list
            print(x_y_coord_list)

            row = x_y_coord_list + [8]  # CHANGE THE DATA LABEL WHEN ADDING NEW DATA

            with open('right_hand_dataset_extended.csv', mode='a', newline='') as csv_file:
                writer = csv.writer(csv_file)
                writer.writerow(row)
                num_data_points += 1
                print(num_data_points)
            if num_data_points == 500:
                print("Done")
                break

        # Hit q to terminate the openpose window, and exit program
        if cv2.waitKey(1) == ord('q'):
            cap.release()
            cv2.destroyAllWindows()
            print("Terminating Program")
            exit()

except Exception as e:
     print(e)
     sys.exit(-1)