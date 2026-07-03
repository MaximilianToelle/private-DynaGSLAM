import torch
import numpy as np
from torch.nn import functional as F
import sys
import os
import cv2
import time
from PIL import Image
import matplotlib.pyplot as plt

from .motion_models.raft import RAFT
from .motion_models.utils import flow_viz
from .motion_models.sam2.build_sam import build_sam2_camera_predictor


def writeFlowFile(filename, uv):
    """
    According to the matlab code of Deqing Sun and c++ source code of Daniel Scharstein
    Contact: dqsun@cs.brown.edu
    Contact: schar@middlebury.edu
    """
    TAG_STRING = np.array(202021.25, dtype=np.float32)
    if uv.shape[2] != 2:
        sys.exit("writeFlowFile: flow must have two bands!")
    H = np.array(uv.shape[0], dtype=np.int32)
    W = np.array(uv.shape[1], dtype=np.int32)
    with open(filename, 'wb') as f:
        f.write(TAG_STRING.tobytes())
        f.write(W.tobytes())
        f.write(H.tobytes())
        f.write(uv.tobytes())
        
def estimate_flow(frame1, frame2, frame_id, sam_predictor=None): #flow estimation vector is from frame2 pointing to frame1
    model = torch.nn.DataParallel(RAFT())
    model.load_state_dict(torch.load('SLAM/multiprocess/motion_models/ckpts/raft-things.pth'))
    model = model.module

    model.to('cuda')
    model.eval()

    image1 = frame1.to('cuda').permute(2,0,1).unsqueeze(0) * 255.
    image2 = frame2.to('cuda').permute(2,0,1).unsqueeze(0) * 255.
    flow_low, flow_up = model(image1, image2, iters=20, test_mode=True)
    model.eval()
    for param in model.parameters():
        param.requires_grad = False

    
    #save flow
    flo = flow_up[0].permute(1, 2, 0).detach().cpu().numpy()
    # save raw flow
    #writeFlowFile(rawflopath[:-4]+'.flo', flo)
    # save image.
    flo = flow_viz.flow_to_image(flo)
    
    
    # Canny edge detector to get the edge
    gray = cv2.cvtColor(flo, cv2.COLOR_BGR2GRAY)

    # (Optional) Blur to reduce noise before Canny
    gray_blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # 2. Edge detection
    edges = cv2.Canny(gray_blur, threshold1=10, threshold2=10)

    # 3. Close gaps in edges
    #    This helps transform broken edges into more complete/closed contours.
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (100, 100))
    closed_edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
    
    
    # Invert edges: edges are white => become black; background black => becomes white
    inverted = cv2.bitwise_not(closed_edges)

    # Prepare a mask for floodFill (note: dimensions must be image + 2)
    h, w = inverted.shape[:2]
    flood_mask = np.zeros((h + 2, w + 2), np.uint8)
    
    # Flood fill from (0, 0) – top-left corner (often background)
    # We paint the background with white (255).
    cv2.floodFill(
        inverted,            # image to be flood-filled (modified in-place)
        flood_mask,          # mask
        (0, 0),              # seed point
        255,                 # newVal: color to fill
        (50,), (50,),          # loDiff, upDiff (tolerance for flood fill) - adjust as needed
        cv2.FLOODFILL_FIXED_RANGE
    )

    # Re-invert so that the hole(s) inside your objects get filled
    mask = cv2.bitwise_not(inverted)
    
    
    # 7. Segment the object: keep only pixels inside the largest contour
    segmented = cv2.bitwise_and(flo, flo, mask=mask)

    
    # Show results
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 4, 1)
    plt.imshow(flo)
    plt.title('Original Image')
    plt.axis('off')

    plt.subplot(1, 4, 2)
    plt.imshow(closed_edges, cmap='gray')
    plt.title('Closed Edges')
    plt.axis('off')

    plt.subplot(1, 4, 3)
    plt.imshow(mask)
    plt.title('Mask')
    plt.axis('off')
    
    plt.subplot(1, 4, 4)
    plt.imshow(segmented)
    plt.title('Segmented Object')
    plt.axis('off')
    
    plt.tight_layout()
    #plt.savefig('//hdd2/output/omd/swinging_4_unconstrained/flowmap/%d.png'%(frame_id), bbox_inches='tight')
    plt.show()

    mask[mask!=0] = 1
    num_labels, labels_im, stats, centroids = cv2.connectedComponentsWithStats(mask)
    
    # Store (area, centroid) for each component (skip background i=0)
    component_info = []
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        centroid = centroids[i]
        component_info.append((area, centroid))

    # Sort by area in descending order
    component_info.sort(reverse=True, key=lambda x: x[0])
    
    # Select top 4 largest components
    top_4_centroids = [info[1] for info in component_info[:4]]
    
    
    
    sam2_checkpoint = "//home/blark/Desktop/2024/qcomm/dynagslam/SLAM/multiprocess/motion_models/ckpts/sam2.1_hiera_small.pt"
    model_cfg = "//home/blark/Desktop/2024/qcomm/dynagslam/configs/omd/sam2.1_hiera_s.yaml"
    #sam2_checkpoint = "//home/blark/Desktop/2024/qcomm/dynagslam/SLAM/multiprocess/motion_models/ckpts/sam2.1_hiera_tiny.pt"
    #model_cfg = "//home/blark/Desktop/2024/qcomm/dynagslam/configs/omd/sam2.1_hiera_t.yaml"
    if sam_predictor is None:
        predictor = build_sam2_camera_predictor(model_cfg, sam2_checkpoint)
    else:
        predictor = sam_predictor
    frame = frame1.detach().cpu().numpy()*255
    frame = frame.astype(np.uint8)
    all_mask_vis = np.zeros((h, w, 1), dtype=np.uint8)
    all_mask = torch.zeros((h, w, 1)).to(torch.bool).cuda()
    if frame_id == 1:
        predictor.load_first_frame(frame)
        ann_frame_idx = 0
        labels = np.array([1], dtype=np.int32)
        for i, point in enumerate(top_4_centroids):
            ann_obj_id = i+1
            _, out_obj_ids, out_mask_logits = predictor.add_new_prompt(
             frame_idx=ann_frame_idx, obj_id=ann_obj_id, points=point.reshape(1, -1), labels=labels)
            out_mask = (out_mask_logits[i] > 0.0).permute(1, 2, 0)
            all_mask = all_mask | out_mask
            
    elif frame_id>1:
        out_obj_ids, out_mask_logits = predictor.track(frame)
        for i in range(0, len(out_obj_ids)):
            out_mask = (out_mask_logits[i] > 0.0).permute(1, 2, 0)
            all_mask = all_mask | out_mask

    all_mask = all_mask.squeeze(-1)
    return flow_up, all_mask, predictor
    

    
    
    
    