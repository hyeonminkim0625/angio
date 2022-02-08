import numpy as  np
import cv2

annotated_dot = np.zeros((512,512))

import numpy as  np
import cv2

def gaussian_heatmap(sigma: int, spread: int):
    extent = int(spread * sigma)
    center = spread * sigma / 2
    heatmap = np.zeros([extent, extent], dtype=np.float32)
    for i_ in range(extent):
        for j_ in range(extent):
            heatmap[i_, j_] = 1 / 2 / np.pi / (sigma ** 2) * np.exp(
                -1 / 2 * ((i_ - center - 0.5) ** 2 + (j_ - center - 0.5) ** 2) / (sigma ** 2))
    heatmap = (heatmap / np.max(heatmap) * 255).astype(np.uint8)
    return heatmap


hm = gaussian_heatmap(sigma=1.6, spread=10)
extent = int(16)
center = 8
print(hm)
annotated_dot[200-center:200+center,200-center:200+center] = hm
cv2.imshow("hm", annotated_dot)
cv2.waitKey(0)