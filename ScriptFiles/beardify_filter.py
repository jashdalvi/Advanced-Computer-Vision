import cv2
import numpy as np
import dlib

def landmarks_to_points(landmarks):
    points = []
    for i in range(len(landmarks.parts())):
        points.append((landmarks.part(i).x,landmarks.part(i).y))

    return points

def warp_triangle(img1,img2,tri_in,tri_out):
    rect1 = cv2.boundingRect(tri_in)
    rect2 = cv2.boundingRect(tri_out)

    img1_cropped = img1[rect1[1]:rect1[1] + rect1[3], rect1[0]:rect1[0] + rect1[2]]
    

    tri_in_cropped = []
    tri_out_cropped = []
    for i in range(3):
        tri_in_cropped.append((int(tri_in[0,i,0]-rect1[0]),int(tri_in[0,i,1] - rect1[1])))
        tri_out_cropped.append((int(tri_out[0,i,0]-rect2[0]),int(tri_out[0,i,1] - rect2[1])))

    tri_in_cropped = np.float32(tri_in_cropped).reshape(3,1,2)
    tri_out_cropped = np.float32(tri_out_cropped).reshape(3,1,2)

    warp_mat = cv2.getAffineTransform(tri_in_cropped,tri_out_cropped)

    img2_cropped = cv2.warpAffine(img1_cropped,warp_mat,(rect2[2],rect2[3]), None, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101 )

    mask = np.zeros_like(img2_cropped)
    mask = cv2.fillConvexPoly(mask,np.int32(tri_out_cropped),(255,255,255), 16, 0)
    mask = np.float32(mask)/255.0
    img2_cropped = img2_cropped*mask
    img2[rect2[1]:rect2[1]+rect2[3], rect2[0]:rect2[0]+rect2[2]] = img2[rect2[1]:rect2[1]+rect2[3], rect2[0]:rect2[0]+rect2[2]] * ( (1.0, 1.0, 1.0) - mask )

    img2[rect2[1]:rect2[1]+rect2[3], rect2[0]:rect2[0]+rect2[2]] = img2[rect2[1]:rect2[1]+rect2[3], rect2[0]:rect2[0]+rect2[2]] + img2_cropped



def calculate_delaunay(img,points):
    rect = (0,0,img.shape[1],img.shape[0])
    subdiv = cv2.Subdiv2D(rect)  
    print(img.shape)
    for p in points:
        subdiv.insert(p)

    triangles = subdiv.getTriangleList()

    indexes = []
    for tri in triangles:
        pt1 = np.float32([tri[0],tri[1]])
        pt2 = np.float32([tri[2],tri[3]])
        pt3 = np.float32([tri[4],tri[5]])

        index1 = np.argmin(np.sqrt(((np.float32(points) - pt1)**2).sum(axis = 1)),axis = 0)
        index2 = np.argmin(np.sqrt(((np.float32(points) - pt2)**2).sum(axis = 1)),axis = 0)
        index3 = np.argmin(np.sqrt(((np.float32(points) - pt3)**2).sum(axis = 1)),axis = 0)

        indexes.append((index1,index2,index3))

    return indexes

beard_image = cv2.imread("../data/images/beard1.png",-1)


b,g,r,a = cv2.split(beard_image)
beard_bgr = cv2.merge([b,g,r])

beard_mask = cv2.merge([a,a,a])


with open("../data/images/beard1.png.txt",'r') as f:
    points = f.read().rstrip("\n").split("\n")

    beard_points = []
    for pt in points:
       x , y = pt.split()
       beard_points.append((int(x),int(y)))

corresponding_points = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
                 31, 32, 33, 34, 35, 55, 56, 57, 58, 59]


dt_index = calculate_delaunay(beard_bgr,beard_points)
landmark_detector_path = "../data/models/shape_predictor_68_face_landmarks.dat"

face_detector = dlib.get_frontal_face_detector()

landmark_detector = dlib.shape_predictor(landmark_detector_path)

cap = cv2.VideoCapture(0)

while cap.isOpened():

    ret,frame = cap.read()

    if not ret:
        break

    frame_rgb = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
    face_rects = face_detector(frame_rgb,0)

    if len(face_rects) == 0:
        continue
    
    new_rect = dlib.rectangle(int(face_rects[0].left()), 
                          int(face_rects[0].top()), 
                          int(face_rects[0].right()), 
                          int(face_rects[0].bottom()))

    landmarks = landmark_detector(frame_rgb,new_rect)

    points = landmarks_to_points(landmarks)

    points_on_beard = []
    for pt in corresponding_points:
        points_on_beard.append(points[pt])
    frame_copy1 = np.zeros_like(frame)
    frame_copy2 = np.zeros_like(frame)
    for idxs in dt_index:
        tri_in = []
        tri_out = []
        for idx in idxs:
            tri_in.append(beard_points[idx])
            tri_out.append(points_on_beard[idx])

        tri_in = np.float32(tri_in).reshape(1,3,2)
        tri_out = np.float32(tri_out).reshape(1,3,2)
        warp_triangle(beard_bgr,frame_copy1,tri_in,tri_out)
        warp_triangle(beard_mask,frame_copy2,tri_in,tri_out)


    mask = np.float32(frame_copy2)/255
    inter1 = np.float32(frame_copy1) * mask
    inter2 = np.float32(frame) * (1 - mask)

    result = cv2.add(inter1,inter2)
    result = np.uint8(result)


    cv2.imshow("Beardify",result)
    k = cv2.waitKey(1)
    if k == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()



