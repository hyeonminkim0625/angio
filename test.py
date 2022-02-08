import numpy as  np
import cv2

annotated_dot = np.zeros((512,512))
annotated_dot[int(200),int(200)]=255# y1 x1
annotated_dot[int(400),int(400)]=255

annotated_dot = cv2.GaussianBlur(annotated_dot,(15,15),0)*20

temp = np.stack([annotated_dot,annotated_dot,annotated_dot],axis=2)
print(annotated_dot[200,200])
cv2.imshow('dfdf',temp)
cv2.waitKey(0)
cv2.destroyAllWindows()