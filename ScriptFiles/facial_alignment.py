import numpy as np
import cv2
import dlib


landmark_detector_path = "../data/models/shape_predictor_5_face_landmarks.dat"

face_detector = dlib.get_frontal_face_detector()

landmark_detector = dlib.shape_predictor(landmark_detector_path)

image = cv2.imread("../data/images/face2.png")
image_rgb = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)

face_rects = face_detector(image_rgb,0)

new_rect = dlib.rectangle(int(face_rects[0].left()), 
                          int(face_rects[0].top()), 
                          int(face_rects[0].right()), 
                          int(face_rects[0].bottom()))

landmarks = landmark_detector(image,new_rect)

width,height  = 600,600
points = []
for i in range(len(landmarks.parts())):
    points.append([landmarks.part(i).x,landmarks.part(i).y])

eye_points_src = [points[2],points[0]]
eye_points_dst = [[int(0.3 * width),int(height/3)],[int(0.7*width),int(height/3)]]

cos_theta = np.cos(60*np.pi/180)
sin_theta = np.sin(60*np.pi/180)

x_src = cos_theta*(eye_points_src[0][0] - eye_points_src[1][0]) - sin_theta*(eye_points_src[0][1] - eye_points_src[1][1]) + eye_points_src[1][0]
y_src = cos_theta*(eye_points_src[0][0] - eye_points_src[1][0]) + sin_theta*(eye_points_src[0][1] - eye_points_src[1][1]) + eye_points_src[1][1]

x_dst = cos_theta*(eye_points_dst[0][0] - eye_points_dst[1][0]) - sin_theta*(eye_points_dst[0][1] - eye_points_dst[1][1]) + eye_points_dst[1][0]
y_dst = cos_theta*(eye_points_dst[0][0] - eye_points_dst[1][0]) + sin_theta*(eye_points_dst[0][1] - eye_points_dst[1][1]) + eye_points_dst[1][1]

eye_points_src.append([x_src,y_src])
eye_points_dst.append([x_dst,y_dst])

eye_points_src = np.int32(eye_points_src).reshape(3,1,2)
eye_points_dst = np.int32(eye_points_dst).reshape(3,1,2)

transform_mat = cv2.estimateAffinePartial2D(eye_points_src,eye_points_dst)[0]

face_aligned = cv2.warpAffine(image,transform_mat,(width,height))

cv2.imshow("face aligned",face_aligned)
cv2.waitKey(0)

cv2.destroyAllWindows()
