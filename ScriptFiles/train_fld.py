import numpy as np
import cv2
import dlib
import imutils
import os


model_name = "shape_predictor_70_face_landmarks.dat"

options = dlib.shape_predictor_training_options()
options.cascade_depth = 10
options.num_trees_per_cascade_level = 500
options.tree_depth = 4
options.nu = 0.1
options.oversampling_amount = 20
options.feature_pool_size = 400
options.feature_pool_region_padding = 0
options.lambda_param = 0.1
options.num_test_splits = 20

# Tell the trainer to print status messages to the console so we can
# see training options and how long the training will take.
options.be_verbose = True

training_xml_path = "/home/jash/Desktop/JashWork/Advanced-Computer-Vision/data/models/facial_landmark_data/70_points/training_with_face_landmarks.xml"
testing_xml_path = "/home/jash/Desktop/JashWork/Advanced-Computer-Vision/data/models/facial_landmark_data/70_points/testing_with_face_landmarks.xml"
output_model_path = "/home/jash/Desktop/JashWork/Advanced-Computer-Vision/data/models/" + model_name

if os.path.exists(training_xml_path) and os.path.exists(testing_xml_path):
    dlib.train_shape_predictor(training_xml_path,output_model_path,options)

    print("Training error: {}".format(dlib.test_shape_predictor(training_xml_path,output_model_path)))
    print("Testing error: {}".format(dlib.test_shape_predictor(testing_xml_path,output_model_path)))

