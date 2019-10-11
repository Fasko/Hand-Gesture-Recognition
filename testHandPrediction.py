'''
# Michael Fasko Jr & Jacob Calfee
# Cleveland State University
# Hackron 4k (10/5/19 - 10/6/19) - Hand Gesture Recognition with Machine Learning

 # Program functions as a demo by using the trained .h5 TensorFlow Keras model for Hand Gesture Recognition.
    By reading webcam frames, the user's right hand data is used as input to make a prediction on the corresponding hand
    gesture from the 9 classes created.
 # The program displays the model's predicted hand symbol's string near the user's right hand after each prediction.

 # NOTE: Pass '--number_people_max 1' as an argument for best efficiency
'''

import argparse
import os
import sys

import cv2
import numpy as np
from keras.models import load_model
from sklearn.preprocessing import Normalizer

# Paths for OpenPose
dir_path = "C:\\Users\\Fasko\\Documents\\Complex-Human-Activity-Recognition\\PoseEvaluation"
sys.path.append("{}\\python\\openpose\\Release".format(dir_path))
os.environ["PATH"] = os.environ["PATH"] + ";{}\\Release;{}\\bin".format(dir_path, dir_path)

try:
    import pyopenpose as op
except ImportError as e:
    print(e, file=sys.stderr)

# OpenPose Flags
parser = argparse.ArgumentParser()
parser.add_argument("--hand", default=True, help="Process an image. Read all standard formats (jpg, png, bmp, etc.).")
parser.add_argument("--display", default=0, help="Process an image. Read all standard formats (jpg, png, bmp, etc.).")

args = parser.parse_known_args()

# OpenPose Custom Params (refer to include/openpose/flags.hpp for more parameters)
params = dict()
params["model_folder"] = "{}\\models".format(dir_path)
params["hand"] = True
params["display"] = 0
# Add others command line arguments
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
    import tensorflow as tf
    # Starting OpenPose
    opWrapper = op.WrapperPython()
    opWrapper.configure(params)
    opWrapper.start()
    datum = op.Datum()

    # Start TensorFlow, load Keras model
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.1  # Only allocates a portion of VRAM to TensorFlow
    session = tf.Session(config=config)
    tf_model = load_model('normalized_epochs85_42_data_points_extended_9_outputs10_06_2019_09_43_22.h5') # 'normalized_epochs200_10_data_points10_06_2019_02_00_54.h5

    # Capture Frames
    cap = cv2.VideoCapture(0)
    num_data_points = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Use Webcam frames, render OpenPose
        datum.cvInputData = frame
        opWrapper.emplaceAndPop([datum])
        op_frame = datum.cvOutputData

        window_name = "Hand Classification Window"

        # All available hand keypoints (OpenPose 1.5 (1-20))
        hand_data = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]

        x_set =[]
        y_set = []
        conf_level_sum = 0

        # Ensure hand keypoints exist before doing classification
        try:
            rightHandKeypoints = datum.handKeypoints[1]
            #print(rightHandKeypoints)
            for entries in rightHandKeypoints:
                for hand_entry in hand_data:
                    conf_level_sum += entries[hand_entry][2]
                    x_set.append(entries[hand_entry][0])
                    y_set.append(entries[hand_entry][1])

            xy_set = x_set + y_set
            #print(xy_set)
            xy_set = np.asarray(xy_set, dtype=np.float32)
            xy_set = xy_set.reshape(1,-1)
            transformer = Normalizer().fit(xy_set)
            X_test = transformer.transform(xy_set)

            predictions = tf_model.predict(xy_set)
            predictions = predictions.flatten()
            prediction_strings = ["Front Fist", "Back Fist", "Front Peace Sign", "Back Peace Sign", "Front Palm", "Back Palm", "Thumbs Up", "Thumbs Down", "Ok Sign"]
        except:
            pass
        i = 0
        if (conf_level_sum/9) > .15:
            print(predictions)
            for x in predictions:
                if x == 1:
                    print(prediction_strings[i])
                    cv2.putText(op_frame,prediction_strings[i],(x_set[8], int(y_set[8] + 10)), cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(0,0,0),thickness=2)
                    cv2.putText(op_frame, prediction_strings[i], (x_set[8], int(y_set[8] + 10)),
                                cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(255, 255, 255), thickness=1)
                elif x == 0:
                    pass
                i += 1

            # Hit q to terminate the openpose window, and exit program
        else:
            cv2.putText(op_frame, "Please put hand in frame", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, fontScale=1,
                        color=(0, 0, 0), thickness=2)
            cv2.putText(op_frame, "Please put hand in frame", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, fontScale=1,
                        color=(255, 255, 255), thickness=1)

        cv2.namedWindow(window_name, cv2.WND_PROP_FULLSCREEN)
        cv2.imshow(window_name, op_frame)

        if cv2.waitKey(1) == ord('q'):
            cap.release()
            cv2.destroyAllWindows()
            print("Terminating Program")
            exit()

except Exception as e:
     print(e)
     sys.exit(-1)
