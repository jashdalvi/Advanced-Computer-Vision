import cv2
import numpy as np
import dlib

def warpTriangle(img1, img2, tri1, tri2) :

  # Find bounding rectangle for each triangle
  r1 = cv2.boundingRect(tri1)
  r2 = cv2.boundingRect(tri2)

  # Crop input image
  img1Cropped = img1[r1[1]:r1[1] + r1[3], r1[0]:r1[0] + r1[2]]

  # Offset points by left top corner of the respective rectangles
  tri1Cropped = []
  tri2Cropped = []

  for i in range(0, 3):
    tri1Cropped.append(((tri1[0][i][0] - r1[0]),(tri1[0][i][1] - r1[1])))
    tri2Cropped.append(((tri2[0][i][0] - r2[0]),(tri2[0][i][1] - r2[1])))

  # Given a pair of triangles, find the affine transform.
  warpMat = cv2.getAffineTransform( np.float32(tri1Cropped), np.float32(tri2Cropped) )

  # Apply the Affine Transform just found to the src image
  img2Cropped = cv2.warpAffine( img1Cropped, warpMat, (r2[2], r2[3]), None, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101 )

  # Get mask by filling triangle
  mask = np.zeros((r2[3], r2[2], 3), dtype = np.float32)
  cv2.fillConvexPoly(mask, np.int32(tri2Cropped), (1.0, 1.0, 1.0), 16, 0)

  img2Cropped = img2Cropped * mask

  # Copy triangular region of the rectangular patch to the output image
  img2[r2[1]:r2[1]+r2[3], r2[0]:r2[0]+r2[2]] = img2[r2[1]:r2[1]+r2[3], r2[0]:r2[0]+r2[2]] * ( (1.0, 1.0, 1.0) - mask )

  img2[r2[1]:r2[1]+r2[3], r2[0]:r2[0]+r2[2]] = img2[r2[1]:r2[1]+r2[3], r2[0]:r2[0]+r2[2]] + img2Cropped

def landmarks_to_points(landmarks):
    points = []
    for i in range(len(landmarks.parts())):
        points.append((landmarks.part(i).x,landmarks.part(i).y))

    return points

ted_image = cv2.imread("../data/images/Devansh02.jpg")
# donald_image = cv2.imread("../data/images/Jash01.pnng")
cap = cv2.VideoCapture(0)



landmark_detector_path = "../data/models/shape_predictor_68_face_landmarks.dat"

face_detector = dlib.get_frontal_face_detector()

landmark_detector = dlib.shape_predictor(landmark_detector_path)

while cap.isOpened():

    ret ,donald_image = cap.read()

    if not ret:
        break
    src_image_rgb = cv2.cvtColor(ted_image,cv2.COLOR_BGR2RGB)
    dst_image_rgb = cv2.cvtColor(donald_image,cv2.COLOR_BGR2RGB)


    face_rects_src = face_detector(src_image_rgb,0)
    face_rects_dst = face_detector(dst_image_rgb,0)

    if len(face_rects_dst) == 0:
        continue

    new_rect_src = dlib.rectangle(int(face_rects_src[0].left()),
                                    int(face_rects_src[0].top()),
                                    int(face_rects_src[0].right()),
                                    int(face_rects_src[0].bottom()))

    new_rect_dst = dlib.rectangle(int(face_rects_dst[0].left()),
                                    int(face_rects_dst[0].top()),
                                    int(face_rects_dst[0].right()),
                                    int(face_rects_dst[0].bottom()))

    landmarks_src = landmark_detector(src_image_rgb,new_rect_src)
    landmarks_dst = landmark_detector(dst_image_rgb,new_rect_dst)

    src_pts = landmarks_to_points(landmarks_src)
    dst_pts = landmarks_to_points(landmarks_dst)

    index = cv2.convexHull(np.array(dst_pts),returnPoints = False)
    hull1 = []
    hull2 = []
    for i in range(len(index)):
        hull1.append(src_pts[index[i,0]])
        hull2.append(dst_pts[index[i,0]])

    height,width = donald_image.shape[:2]
    rect = (0,0,width,height)
    subdiv = cv2.Subdiv2D(rect)

    for p in hull2:
        subdiv.insert(p)

    triangle_list = subdiv.getTriangleList()
    indexes = []
    for tri in triangle_list:
        pt1 = (tri[0],tri[1])
        pt2 = (tri[2],tri[3])
        pt3 = (tri[4],tri[5])

        pt1_index = np.argmin(np.sqrt(((np.float32(hull2) - np.float32(pt1))**2).sum(axis = 1)),axis = 0)
        pt2_index = np.argmin(np.sqrt(((np.float32(hull2) - np.float32(pt2))**2).sum(axis = 1)),axis = 0)
        pt3_index = np.argmin(np.sqrt(((np.float32(hull2) - np.float32(pt3))**2).sum(axis = 1)),axis = 0)

        indexes.append((pt1_index,pt2_index,pt3_index))

    donald_image_copy = donald_image.copy()
    for i in range(len(indexes)):
        tri_in = []
        tri_out = []
        for idx in indexes[i]:
            tri_in.append(hull1[idx])
            tri_out.append(hull2[idx])

        tri1 = np.float32(tri_in).reshape(1,3,2)
        tri2 = np.float32(tri_out).reshape(1,3,2)

        warpTriangle(ted_image,donald_image_copy, tri1, tri2)

    mask = np.zeros_like(donald_image)
    cv2.fillConvexPoly(mask,np.int32(hull2).reshape(-1,1,2),(255,255,255))

    rect_bound = cv2.boundingRect(np.int32(hull2).reshape(-1,1,2))
    center = (rect_bound[0] + rect_bound[2]//2,rect_bound[1] + rect_bound[3]//2)

    face_swap_output = cv2.seamlessClone(donald_image_copy,donald_image,mask,center,cv2.NORMAL_CLONE)

    cv2.imshow("Face Swap",face_swap_output)
    k = cv2.waitKey(1)
    if k == ord('q'):
        break
# cv2.imwrite("../data/images/face_swap1.jpg",face_swap_output)
cv2.destroyAllWindows()
    
    

