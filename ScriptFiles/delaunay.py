import numpy as np
import cv2
import dlib

def get_landmark_points(landmarks):
    points = []
    for i in range(len(landmarks.parts())):
        points.append((landmarks.part(i).x,landmarks.part(i).y))

    return points

def draw_circles(frame,points):
    for i in range(len(points)):
        cv2.circle(frame,(points[i][0],points[i][1]),5,(0,0,255),-1)


shape_predictor_path = "../data/models/shape_predictor_68_face_landmarks.dat"

face_detector = dlib.get_frontal_face_detector()
landmark_detector = dlib.shape_predictor(shape_predictor_path)
cap = cv2.VideoCapture(0)

while True:

    ret,frame = cap.read()
    if not ret:
        break

    frame_rgb = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
    frame_resize = cv2.resize(frame,(320,240),interpolation = cv2.INTER_LINEAR)
    resize_height_scale = frame.shape[0]/240
    resize_width_scale = frame.shape[1]/320
    frame_resize_rgb = cv2.cvtColor(frame_resize,cv2.COLOR_BGR2RGB)

    face_rects = face_detector(frame_resize_rgb,0)

    for i in range(len(face_rects)):
        new_rect = dlib.rectangle(int(face_rects[i].left()*resize_width_scale),
                                int(face_rects[i].top()*resize_height_scale),
                                int(face_rects[i].right()*resize_width_scale),
                                int(face_rects[i].bottom()*resize_height_scale))

        x1 = int(face_rects[i].left()*resize_width_scale)
        y1 = int(face_rects[i].top()*resize_height_scale)
        x2 = int(face_rects[i].right()*resize_width_scale)
        y2 = int(face_rects[i].bottom()*resize_height_scale)
        landmarks = landmark_detector(frame_rgb,new_rect)

        points = get_landmark_points(landmarks)

        draw_circles(frame,points)

        rect = (0,0,frame.shape[1],frame.shape[0])

        subdiv = cv2.Subdiv2D(rect)
        subdiv.insert((x1,y1))
        subdiv.insert((x2,y2))
        subdiv.insert((x1,y2))
        subdiv.insert((x2,y1))
        cv2.rectangle(frame,(x1,y1),(x2,y2),(0,0,255),2,cv2.LINE_AA)

        for p in points:
            subdiv.insert(p)

        triangle_list = subdiv.getTriangleList()
        for tri in triangle_list:
            tri_points = []
            for i in range(0,len(tri),2):
                tri_points.append((tri[i],tri[i+1]))

            tri_points = np.array(tri_points).reshape(-1,3,2).astype(np.int32)
            cv2.polylines(frame,[tri_points],True, (0,255,0), thickness=2, lineType=cv2.LINE_8)

    cv2.imshow("frame",frame)
    k = cv2.waitKey(1)
    if k == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()  