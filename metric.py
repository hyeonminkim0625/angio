import numpy as np
from sklearn.metrics import pairwise_distances
import cv2
import torch

def averaged_hausdorff_distance(set1, set2, max_ahd=np.inf):
    """
    Compute the Averaged Hausdorff Distance function
    between two unordered sets of points (the function is symmetric).
    Batches are not supported, so squeeze your inputs first!
    :param set1: Array/list where each row/element is an N-dimensional point.
    :param set2: Array/list where each row/element is an N-dimensional point.
    :param max_ahd: Maximum AHD possible to return if any set is empty. Default: inf.
    :return: The Averaged Hausdorff Distance between set1 and set2.
    """

    if len(set1) == 0 or len(set2) == 0:
        return max_ahd

    set1 = np.array(set1)
    set2 = np.array(set2)

    assert set1.ndim == 2, 'got %s' % set1.ndim
    assert set2.ndim == 2, 'got %s' % set2.ndim

    assert set1.shape[1] == set2.shape[1], \
        'The points in both sets must have the same number of dimensions, got %s and %s.'\
        % (set2.shape[1], set2.shape[1])

    d2_matrix = pairwise_distances(set1, set2, metric='euclidean')

    res = np.average(np.min(d2_matrix, axis=0)) + \
        np.average(np.min(d2_matrix, axis=1))

    return res

"""
pred, target : probability numpy mask
(num_classes+1, H, W)
return [class 1_iou,,,, class n_iou]
"""
def calculate_iou(pred,target,num_classes):

    #pred_mask = np.argmax(pred,axis=0)
    #target_mask = np.argmax(target,axis=0)
    iou_list = []
    for i in range(1,num_classes):
        iou_score = (torch.sum((pred[i]==True)&(target[i]==True))+ 1e-6) /(torch.sum((pred[i]==True)|(target[i]==True))+ 1e-6)
        iou_list.append(iou_score)
    
    return iou_list
    
"""

"""
def compute_mean_iou(pred, label):
    
    unique_labels = np.unique(label)
    num_unique_labels = len(unique_labels);

    I = np.zeros(num_unique_labels)
    U = np.zeros(num_unique_labels)

    for index, val in enumerate(unique_labels):
        pred_i = pred == val
        label_i = label == val

        I[index] = float(np.sum(np.logical_and(label_i, pred_i)))
        U[index] = float(np.sum(np.logical_or(label_i, pred_i)))


    mean_iou = np.mean(I / U)
    return mean_iou

def calculate_overlab_contour(img):
    img = img.astype(np.float32)
    imgray = 255 - img*255
    imgray = np.stack([imgray,imgray,imgray],axis=2)
    imgray = cv2.cvtColor(imgray, cv2.COLOR_BGR2GRAY)
    ret, imthres = cv2.threshold(imgray, 127, 255, cv2.THRESH_BINARY_INV)

    contour, hi = cv2.findContours(imthres.astype(np.uint8), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)

    contour_mask = np.zeros((256,256))
    for i in contour:
        for k in i:
            contour_mask[tuple(k[0])] = 1
    
    return contour_mask

"""
Boundary IOU
(G_d & G) & (P_d & P)
---------------------
(G_d & G) | (P_d & P)
boundary을 기준으로 한 d 필셀 중 boundary 안쪽을 기준으로 사용하겠다
"""

def boundary_iou():
    pass