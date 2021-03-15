import numpy as np
import cv2

bg = cv2.imread("../data/images/backGroundLarge.jpg")
fg_with_mask = cv2.imread("../data/images/foreGroundAssetLarge.png",-1)

height, width = bg.shape[:2]
fg_with_mask = cv2.resize(fg_with_mask,(width,height),interpolation=cv2.INTER_CUBIC)

fg = fg_with_mask[:,:,:3]
alpha_mask = fg_with_mask[:,:,-1]


alpha_mask = cv2.merge([alpha_mask,alpha_mask,alpha_mask])

alpha_mask = np.float32(alpha_mask)/255.0

bg_new = (bg*(1 - alpha_mask)).astype(np.uint8)
fg_new = (fg*alpha_mask).astype(np.uint8)

output_image = cv2.add(bg_new,fg_new)

output_image = cv2.resize(output_image,None,fx=0.6,fy=0.6,interpolation = cv2.INTER_CUBIC)

cv2.imshow("Alpha Blending",output_image)

cv2.waitKey(0)

cv2.destroyAllWindows()


