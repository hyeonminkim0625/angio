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

def gaussian_heatmap_re(heatmap,x,y):
    for i_ in range(512):
        for j_ in range(512):
            heatmap[i_, j_] += ((x-i_)**2 + (y-j_)**2)**0.2
    return heatmap

annotated_dot = np.zeros((512,512))
annotated_dot = gaussian_heatmap_re(annotated_dot,200,200)
annotated_dot = gaussian_heatmap_re(annotated_dot,300,300)
annotated_dot = (annotated_dot / np.max(annotated_dot) * 255).astype(np.uint8)
annotated_dot = 255-annotated_dot

annotated_dot = annotated_dot.astype(np.uint8)
print(annotated_dot[100,100])
print(annotated_dot[80,80])

isotropicGaussianHeatmapImage = cv2.applyColorMap(annotated_dot, 
                                                  cv2.COLORMAP_JET)


cv2.imshow("hm", isotropicGaussianHeatmapImage)
cv2.waitKey(0)