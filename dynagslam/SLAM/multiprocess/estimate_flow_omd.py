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
        
def estimate_flow(frame1, frame2, frame_id): #flow estimation vector is from frame2 pointing to frame1
    model = torch.nn.DataParallel(RAFT())
    model.load_state_dict(torch.load('//home/fawad/ReplaceGSW/gsplat_policy/submodules/DynaGSLAM_official/dynagslam/SLAM/multiprocess/motion_models/raft-things.pth'))
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
    flo = flow_viz.flow_to_image(flo)
    ##if not os.path.exists('//hdd2/output/omd/swinging_4_unconstrained/flowmap'):
    ##    os.makedirs('//hdd2/output/omd/swinging_4_unconstrained/flowmap')
    #cv2.imwrite('//hdd2/output/omd/swinging_4_unconstrained/flowmap/%d.png'%(frame_id), flo[:, :, [2, 1, 0]])

    
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

    '''
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
    '''

    '''
    if not os.path.exists('//hdd2/output/omd/swinging_4_unconstrained/flowmap/frame_%04d'%(frame_id)):
        os.makedirs('//hdd2/output/omd/swinging_4_unconstrained/flowmap/frame_%04d'%(frame_id))
    cv2.imwrite('//hdd2/output/omd/swinging_4_unconstrained/flowmap/frame_%04d/flow.png'%(frame_id), flo[:, :, [2, 1, 0]]) #flo_edge[:, :, [2, 1, 0]])
    cv2.imwrite('//hdd2/output/omd/swinging_4_unconstrained/flowmap/frame_%04d/closed_edge.png'%(frame_id), closed_edges)
    cv2.imwrite('//hdd2/output/omd/swinging_4_unconstrained/flowmap/frame_%04d/mask.png'%(frame_id), mask)
    cv2.imwrite('//hdd2/output/omd/swinging_4_unconstrained/flowmap/frame_%04d/segmented.png'%(frame_id), segmented)'''
    
    
    '''
    # visualize the warped image using est flow to verify the flow is from which frame to which frame 
    # Get the grid of normalized coordinates
    height, width = flow_up.shape[2:]
    y_grid, x_grid = torch.meshgrid(torch.arange(height), torch.arange(width), indexing='ij')
    y_grid = y_grid.cuda()
    x_grid = x_grid.cuda()

    # Normalize the grid to range [-1, 1] for PyTorch grid_sample
    x_grid = (x_grid.float() / (width - 1)) * 2 - 1  # Normalize to [-1, 1]
    y_grid = (y_grid.float() / (height - 1)) * 2 - 1  # Normalize to [-1, 1]

    # Create a grid of shape [480, 640, 2]
    grid = torch.stack((x_grid, y_grid), dim=-1)  # Shape: [480, 640, 2]

    # Add the flow to the grid (flow is in pixel coordinates, so we normalize it)
    flow_normalized = torch.zeros_like(flow_up)
    flow_normalized[:,0,:,:] = flow_up[:,0,:,:] / (width - 1) * 2  # Normalize flow in x direction
    flow_normalized[:,1,:,:] = flow_up[:,1,:,:] / (height - 1) * 2  # Normalize flow in y direction

    new_grid = grid + flow_normalized.permute(0,2,3,1)  # Apply the flow to the grid

    # Reshape the grid for grid_sample: [1, 480, 640, 2]
    #new_grid = new_grid.unsqueeze(0)

    # Warp the image using grid_sample
    warped_image = F.grid_sample(image2, new_grid, mode='bilinear', padding_mode='border', align_corners=True)

    # Convert the image back to [480, 640, 3]
    warped_image = warped_image.squeeze(0).permute(1, 2, 0)

    #warped_image = warped_image.clamp(0, 1)  # Ensure pixel values are in the valid range
    np_image = (warped_image.detach().cpu().numpy()).astype(np.uint8)  # Convert to uint8 format

    # Convert the numpy array to a PIL image and save it
    pil_image = Image.fromarray(np_image)
    pil_image.save('warped.png')
    '''
    mask[mask!=0] = 1
    mask = torch.from_numpy(mask).to(flow_up.device).bool()
    return flow_up, mask
    

    
    
    
    