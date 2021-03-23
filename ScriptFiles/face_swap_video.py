import cv2
import numpy as np
import dlib
import math

def correctColours(im1, im2, points):
    
  blurAmount = 0.5 * np.linalg.norm(np.array(points)[38] - np.array(points)[43])
  blurAmount = int(blurAmount)

  if blurAmount % 2 == 0:
    blurAmount += 1
  
  im1Blur = cv2.blur(im1, (blurAmount, blurAmount), 0)
  im2Blur = cv2.blur(im2, (blurAmount, blurAmount), 0)
  
  # Avoid divide-by-zero errors.
  im2Blur += (2 * (im2Blur <= 1)).astype(im2Blur.dtype)
  
  ret = np.uint8((im2.astype(np.float32) * im1Blur.astype(np.float32) /
                                              im2Blur.astype(np.float32)).clip(0,255))
  return ret

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
        points.append((int(landmarks.part(i).x),int(landmarks.part(i).y)))

    return points

src_image = cv2.imread("../data/images/Devansh01.jpg")

src_image = cv2.resize(src_image,(640,480),interpolation= cv2.INTER_LINEAR)

face_detector = dlib.get_frontal_face_detector()
landmark_detector_path = "../data/models/shape_predictor_68_face_landmarks.dat"

landmark_detector = dlib.shape_predictor(landmark_detector_path)

src_image_rgb = cv2.cvtColor(src_image,cv2.COLOR_BGR2RGB)
face_rects_src = face_detector(src_image_rgb,0)

new_rect_src = dlib.rectangle(int(face_rects_src[0].left()),
                            int(face_rects_src[0].top()),
                            int(face_rects_src[0].right()),
                            int(face_rects_src[0].bottom()))

landmarks_src = landmark_detector(src_image_rgb,new_rect_src)

points_src = landmarks_to_points(landmarks_src)

indexes_src = cv2.convexHull(np.array(points_src),returnPoints = False)
add_points_src = [[48],[49],[50],[51],[52],[53],[54],[55],[56],[57],[58]]
combined_indexes_src = np.vstack([np.array(indexes_src),np.array(add_points_src)])
hull1 = []
for i in range(len(combined_indexes_src)):
    hull1.append(points_src[combined_indexes_src[i,0]])

rect = (0,0,src_image.shape[1],src_image.shape[0])
subdiv = cv2.Subdiv2D(rect)

for p in hull1:
    subdiv.insert(p)

indexes_src_hull1 = []
triangle_list = subdiv.getTriangleList()
for tri in triangle_list:
    pt1 = (tri[0],tri[1])
    pt2 = (tri[2],tri[3])
    pt3 = (tri[4],tri[5])

    pt1_index = np.argmin(np.sqrt(((np.float32(hull1) - np.float32(pt1))**2).sum(axis = 1)),axis = 0)
    pt2_index = np.argmin(np.sqrt(((np.float32(hull1) - np.float32(pt2))**2).sum(axis = 1)),axis = 0)
    pt3_index = np.argmin(np.sqrt(((np.float32(hull1) - np.float32(pt3))**2).sum(axis = 1)),axis = 0)

    indexes_src_hull1.append((pt1_index,pt2_index,pt3_index))

cap = cv2.VideoCapture(0)
first_frame = True
while cap.isOpened():

    ret ,dst_image = cap.read()

    if not ret:
        break

    dst_image_rgb = cv2.cvtColor(dst_image,cv2.COLOR_BGR2RGB)
    face_rects_dst = face_detector(dst_image_rgb,0)

    if len(face_rects_dst) == 0:
        continue
    new_rect_dst = dlib.rectangle(int(face_rects_dst[0].left()),
                                int(face_rects_dst[0].top()),
                                int(face_rects_dst[0].right()),
                                int(face_rects_dst[0].bottom()))

    landmarks_dst = landmark_detector(dst_image_rgb,new_rect_dst)

    points_dst = landmarks_to_points(landmarks_dst)

    hull2 = []
    for i in range(len(combined_indexes_src)):
        hull2.append(points_dst[combined_indexes_src[i,0]])

    if first_frame:
        prev_frame_gray = cv2.cvtColor(dst_image_rgb,cv2.COLOR_BGR2GRAY)
        prev_hull = hull2
        first_frame = False
    
    frame_gray = cv2.cvtColor(dst_image_rgb,cv2.COLOR_RGB2GRAY)
    lk_params = dict( winSize  = (101,101),maxLevel = 15,criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.001))
    hull2Next, st , err = cv2.calcOpticalFlowPyrLK(prev_frame_gray,frame_gray,np.array(prev_hull,np.float32), np.array(hull2,np.float32),**lk_params)

    if (len(hull1) > len(hull2)) or len(hull2)!=len(hull2Next):
        continue


    for k in range(0,len(hull2)):
        d = cv2.norm(np.array(hull2[k]) - hull2Next[k])
        alpha = math.exp(-d*d/400)
        hull2[k] = (1 - alpha) * np.array(hull2[k]) + alpha * hull2Next[k]

    
    dst_image_copy1 = dst_image.copy()
    for idxs in indexes_src_hull1:
        tri_in = []
        tri_out = []
        for idx in idxs:
            tri_in.append(hull1[idx])
            tri_out.append(hull2[idx])
        tri_in = np.float32(tri_in).reshape(1,3,2)
        tri_out = np.float32(tri_out).reshape(1,3,2)
        warpTriangle(src_image,dst_image_copy1, tri_in, tri_out)

    output = correctColours(src_image, dst_image_copy1, points_dst)

    # Create a Mask around the face
    re = cv2.boundingRect(np.array(hull2,np.float32))
    centerx = (re[0]+(re[0]+re[2]))/2
    centery = (re[1]+(re[1]+re[3]))/2

    hull3 = []
    for i in range(0,len(hull2)-len(add_points_src)):
    # Take the points just inside of the convex hull
        hull3.append((0.95*(hull2[i][0] - centerx) + centerx, 0.95*(hull2[i][1] - centery) + centery))

    mask1 = np.zeros((dst_image.shape[0], dst_image.shape[1],3), dtype=np.float32)
    hull3Arr = np.array(hull3,np.int32)

    cv2.fillConvexPoly(mask1,hull3Arr,(255.0,255.0,255.0),16,0)

    # Blur the mask before blending
    mask1 = cv2.GaussianBlur(mask1,(51,51),10)

    mask2 = (255.0,255.0,255.0) - mask1

    # cv2.imshow("mask1", np.uint8(mask1))
    # cv2.imshow("mask2", np.uint8(mask2))

    # Perform alpha blending of the two images
    temp1 = np.multiply(output,(mask1*(1.0/255)))
    temp2 = np.multiply(dst_image,(mask2*(1.0/255)))
    result = temp1 + temp2

    cv2.imshow("temp1", np.uint8(temp1))
    cv2.imshow("temp2", np.uint8(temp2))

    result = np.uint8(result)

    prev_frame_gray = frame_gray.copy()
    prev_hull = hull2

    cv2.imshow("After Blending", result)
    if cv2.waitKey(1) == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()


    

        

    






