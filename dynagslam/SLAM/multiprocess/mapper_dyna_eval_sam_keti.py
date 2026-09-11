import os
import shutil
import time
import random
import copy
import torch.multiprocessing as mp
from torch.utils.tensorboard import SummaryWriter
import torchvision
from tqdm import tqdm
from collections import deque
from dynagslam.scene.cameras import Camera
from dynagslam.SLAM.gaussian_pointcloud import *
from dynagslam.SLAM.render import Renderer
from dynagslam.SLAM.utils import merge_ply, rot_compare, trans_compare, bbox_filter
from dynagslam.utils.loss_utils import l1_loss, l2_loss, ssim, psnr
from cuda_utils._C import accumulate_gaussian_error
from dynagslam.utils.monitor import Recorder

import matplotlib.pyplot as plt
import open3d as o3d
from .estimate_flow_omd import estimate_flow
#from .estimate_flow_omd_sam2 import estimate_flow

def save_flo_file(flow, filename):
    """
    Save an optical flow field to a .flo file in Middlebury format.
    
    Args:
        flow (numpy.ndarray): Optical flow array of shape (height, width, 2).
        filename (str): Path to save the .flo file.
    """
    if flow.ndim != 3 or flow.shape[2] != 2:
        raise ValueError("Flow must have shape (height, width, 2)")
    
    height, width = flow.shape[:2]
    
    with open(filename, 'wb') as f:
        np.array([202021.25], dtype=np.float32).tofile(f)  # Magic number
        np.array([width, height], dtype=np.int32).tofile(f)  # Width and height
        flow.astype(np.float32).tofile(f)  # Flow data


class Mapping(object):
    def __init__(self, args, recorder=None) -> None:
        self.temp_pointcloud = GaussianPointCloud(args)
        self.pointcloud = GaussianPointCloud(args)
        self.stable_pointcloud = GaussianPointCloud(args)
        self.dyna_pointcloud = GaussianPointCloud(args)
        self.dyna_pointcloud_future = GaussianPointCloud(args)
        self.recorder = recorder

        self.renderer = Renderer(args)
        self.optimizer = None
        self.time = 0
        self.iter = 0
        self.gaussian_update_iter = args.gaussian_update_iter
        self.gaussian_update_frame = args.gaussian_update_frame
        self.final_global_iter = args.final_global_iter
        
        self.psnr_pred = []
        self.psnr_pred_dyna = []
        
        # # history management
        self.memory_length = args.memory_length
        self.optimize_frames_ids = []
        self.processed_frames = deque(maxlen=self.memory_length)
        self.processed_map = deque(maxlen=self.memory_length)
        self.dyna_masks = deque(maxlen=self.memory_length)
        self.depth_highgrad_masks = deque(maxlen=self.memory_length)
        self.keyframe_ids = []
        self.keyframe_list = []
        self.keymap_list = []
        self.global_keyframe_num = args.global_keyframe_num
        self.keyframe_trans_thes = args.keyframe_trans_thes
        self.keyframe_theta_thes = args.keyframe_theta_thes
        self.KNN_num = args.KNN_num
        self.KNN_threshold = args.KNN_threshold
        self.history_merge_max_weight = args.history_merge_max_weight
        self.dyna_sample_mask = None
        
        
        # points adding parameters
        self.uniform_sample_num = args.uniform_sample_num #1000000
        self.add_depth_thres = args.add_depth_thres
        self.add_normal_thres = args.add_normal_thres
        self.add_color_thres = args.add_color_thres
        self.add_transmission_thres = args.add_transmission_thres

        self.transmission_sample_ratio = args.transmission_sample_ratio
        self.error_sample_ratio = args.error_sample_ratio
        self.stable_confidence_thres = args.stable_confidence_thres
        self.unstable_time_window = args.unstable_time_window

        # all map shape is [H, W, C], please note the raw image shape is [C, H, W]
        self.min_depth, self.max_depth = args.min_depth, args.max_depth
        self.depth_filter = args.depth_filter
        self.frame_map = {
            "depth_map": torch.empty(0),
            "color_map": torch.empty(0),
            "normal_map_c": torch.empty(0),
            "normal_map_w": torch.empty(0),
            "vertex_map_c": torch.empty(0),
            "vertex_map_w": torch.empty(0),
            "confidence_map": torch.empty(0),
        }
        self.model_map = {
            "render_color": torch.empty(0),
            "render_depth": torch.empty(0),
            "render_normal": torch.empty(0),
            "render_color_index": torch.empty(0),
            "render_depth_index": torch.empty(0),
            "render_transmission": torch.empty(0),
            "confidence_map": torch.empty(0),
        }

        # parameters for eval
        self.save_path = args.save_path
        self.save_step = args.save_step
        self.verbose = args.verbose
        self.mode = args.mode
        self.dataset_type = args.type
        assert self.mode == "single process" or self.mode == "multi process"
        self.use_tensorboard = args.use_tensorboard
        self.tb_writer = None

        self.feature_lr_coef = args.feature_lr_coef
        self.scaling_lr_coef = args.scaling_lr_coef
        self.rotation_lr_coef = args.rotation_lr_coef

        #online sam2
        self.sam_predictor = None
        
        #evaluation
        self.add_ratio_all = []
        self.dyna_gaussian_num = []
        
        
    def mapping(self, frame, frame_eval, frame_map, frame_id, optimization_params, dyna_mask, dyna_mask_eval, flow_gt, t_curr, t_past):

        # Get the high-grad depth mask
        depth_y = frame_map['depth_map'].squeeze(-1).diff(dim=0)
        depth_x = frame_map['depth_map'].squeeze(-1).diff(dim=1)
        # Align dimensions by cropping the larger dimension
        min_height = min(depth_y.shape[0], depth_x.shape[0])  # Handle height alignment
        min_width = min(depth_y.shape[1], depth_x.shape[1])   # Handle width alignment
        # Crop both gradients to ensure alignment
        depth_y_cropped = depth_y[:min_height, :min_width]
        depth_x_cropped = depth_x[:min_height, :min_width]
        depth_gradient_mask = torch.sqrt(depth_x_cropped**2 + depth_y_cropped**2)
        depth_highgrad_mask_temp = depth_gradient_mask > 0.1
        depth_highgrad_mask = torch.ones((depth_highgrad_mask_temp.shape[0]+1, depth_highgrad_mask_temp.shape[1]+1)).bool().to(depth_highgrad_mask_temp.device)
        ##in the above code, the temp will make sure that the last row and column will be 1 rather than 0,why?
        depth_highgrad_mask[:-1, :-1] = depth_highgrad_mask_temp
        self.depth_highgrad_masks.append(depth_highgrad_mask)
        
        dyna_mask = torch.from_numpy(dyna_mask).cuda()
        
        self.frame_map = frame_map
        self.processed_frames.append(frame)
        self.processed_map.append(frame_map)
        self.frame_eval = frame_eval
        '''
        # just for saving .flo
        if frame_id>0:
            flow_forward, _ = estimate_flow(self.processed_map[-2]['color_map'], self.processed_map[-1]['color_map'], frame_id)'''
        
        
        if frame_id>0 and self.dyna_sample_mask is not None:
            # the est flow is pointing from t to t-1!!!!
            with torch.no_grad():
                # t to t-1 from the frame t
                # use well-segmented sam2
                flow_backward, _ = estimate_flow(self.processed_map[-1]['color_map'], self.processed_map[-2]['color_map'], frame_id)
                #use online sam2
                '''
                if self.sam_predictor is None:
                    flow_backward, dyna_mask, self.sam_predictor = estimate_flow(self.processed_map[-1]['color_map'], self.processed_map[-2]['color_map'], frame_id)
                else:
                    flow_backward, dyna_mask, self.sam_predictor = estimate_flow(self.processed_map[-1]['color_map'], self.processed_map[-2]['color_map'], frame_id, self.sam_predictor)
                '''
                self.dyna_masks.append(dyna_mask)
                flow_mask_curr = dyna_mask
                flow_mask_past = self.dyna_sample_mask
                height, width = flow_backward.shape[2:]
                y_grid, x_grid = torch.meshgrid(torch.arange(height), torch.arange(width), indexing='ij')
                y_grid = y_grid.cuda()
                x_grid = x_grid.cuda()
                # Create a grid of shape [480, 640, 2]
                grid = torch.stack((x_grid, y_grid), dim=-1)  # Shape: [480, 640, 2]
                flow_backward = flow_backward.squeeze(0).permute(1,2,0)
                grid_backward = grid + flow_backward
                grid_backward_masked_float = grid_backward[flow_mask_curr]
                grid_backward_masked_int = torch.floor(grid_backward_masked_float).to(torch.int)
                grid_past_masked = grid[flow_mask_past]
                test = np.zeros((480,640,1)).astype(np.float64)
                grid_past_masked = grid_past_masked[:, [1,0]].detach().cpu().numpy()
                grid_backward_masked_int = grid_backward_masked_int[:, [1,0]].detach().cpu().numpy()
                grid_curr_masked = grid[flow_mask_curr]
                grid_curr_masked =  torch.floor(grid_curr_masked[:, [1,0]]).to(torch.int).detach().cpu().numpy()
                '''
                test[grid_past_masked[:, 0], grid_past_masked[:, 1]] = 1.0 # mask in t-1 hehe
                test[grid_curr_masked[:, 0], grid_curr_masked[:, 1]] = 0.5 # mask in t fuck
                test[grid_backward_masked_int[:, 0], grid_backward_masked_int[:, 1]] = 0.1 # warped mask of t-1 to t
                plt.imshow(test, cmap='gray')
                plt.show()'''

                pts_curr = frame_map['vertex_map_w'][grid_curr_masked[:, 0], grid_curr_masked[:, 1], :] ## going from pixel to dynamic point in 3D  
                pts_past = self.dyna_pointcloud._xyz[:, 0, :]
                K = frame.get_intrinsic.cuda()
                fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
                pose_gt = self.processed_frames[-2].get_c2w.cuda()
                grid_tmp = grid_backward.clone()
                grid_tmp[:,:,0] = grid_tmp[:,:,0]/((grid_tmp.shape[1]-1)/2)-1 ##nomalize the grid to [-1, 1] for grid_sample
                grid_tmp[:,:,1] = grid_tmp[:,:,1]/((grid_tmp.shape[0]-1)/2)-1
                depth_interp = F.grid_sample(self.processed_map[-2]['depth_map'].permute(2,0,1).unsqueeze(0), grid_tmp.unsqueeze(0), mode='bilinear', align_corners=True).squeeze(0).permute(1,2,0)
                depth_interp = depth_interp[flow_mask_curr]
                pts_warp = torch.stack([(grid_backward_masked_float[:, 0]-cx)/fx, (grid_backward_masked_float[:, 1]-cy)/fy, torch.ones((grid_backward_masked_float.shape[0])).cuda()], -1)*depth_interp 
                #pts_warp[i]; is the predicted previous-frame 3D position corresponding to current dynamic pixel i
                pts_warp = ((pose_gt @ (torch.cat([pts_warp, torch.ones((pts_warp.shape[0],1)).cuda()], -1)).T).T)[:, :3]

                
                #knn_3d
                nn_dist, indices_past_continue, coord_past = knn_points(
                pts_warp.unsqueeze(0),
                pts_past.unsqueeze(0),
                norm=2,
                K=1, #topk=3
                return_nn=True,
                )
                
                
                nn_dist_mean = torch.mean(nn_dist)
                pts_add_map = nn_dist>0.05*nn_dist_mean # the more robust of the 2D pixel tracker (flow), the higher scale can be set 
                add_ratio = pts_add_map.sum()/len(pts_add_map.flatten())
                self.add_ratio_all.append(add_ratio)
  ##            pts_add_map = torch.ones_like(pts_add_map).bool() #makes evey currrent pixel a new dyna
                mask_past_die = torch.ones((pts_past.shape[0])).bool().cuda()
                indices_past_continue_fine = indices_past_continue.squeeze(0).squeeze(-1)[~pts_add_map.squeeze(0).squeeze(-1)]

                unique_past_indices, match_counts = torch.unique(
                    indices_past_continue_fine,
                    return_counts=True,
                )
                duplicated_past_num = (match_counts > 1).sum().item()
                extra_match_num = torch.clamp(match_counts - 1, min=0).sum().item()
                max_matches_per_past = (
                    match_counts.max().item() if match_counts.numel() > 0 else 0
                )
                print(
                    "KNN many-to-one check:",
                    "accepted matches =", indices_past_continue_fine.numel(),
                    "unique past Gaussians =", unique_past_indices.numel(),
                    "past Gaussians matched multiple times =", duplicated_past_num,
                    "extra duplicate assignments =", extra_match_num,
                    "max matches to one past Gaussian =", max_matches_per_past,
                )

                mask_past_die[indices_past_continue_fine] = False
                object_id = self.dyna_pointcloud.get_object_id
                robot_mask = (object_id >= 1) & (object_id <= 16)

                # Project past Gaussian centers into the CURRENT camera.
                world_to_camera = torch.linalg.inv(frame.get_c2w).to(pts_past)
                pts_camera = (
                    pts_past @ world_to_camera[:3, :3].T
                    + world_to_camera[:3, 3]
                )

                x, y, z = pts_camera.unbind(dim=-1)
                z_safe = z.clamp_min(1e-6)

                # fx, fy, cx, cy, height, width are already defined above.
                u = fx * x / z_safe + cx
                v = fy * y / z_safe + cy

                inside_view = (
                    (z > 0)
                    & (u >= 0) & (u < width)
                    & (v >= 0) & (v < height)
                )

                depth_map = frame_map["depth_map"].squeeze(-1).to(pts_past)

                # Sample depth only at valid projected pixels.
                indices_inside = torch.where(inside_view)[0]
                u_inside = u[indices_inside].long()
                v_inside = v[indices_inside].long()
                depth_observed = depth_map[v_inside, u_inside]

                depth_margin = 0.02  # 2 cm tolerance for depth noise
                valid_depth = torch.isfinite(depth_observed) & (depth_observed > 0)

                visible = torch.zeros_like(inside_view)
                visible[indices_inside] = (
                    valid_depth
                    & (z[indices_inside] <= depth_observed + depth_margin)
                )

                mask_past_die &= visible | robot_mask

                # Delete only unmatched Gaussians inside the current view.
               ## mask_past_die &= inside_view

                t_pred = frame_eval.timestamp
                pts_add_map_total = torch.zeros_like(flow_mask_curr).bool().cuda().flatten()
                pts_add_map_total[torch.nonzero(flow_mask_curr.flatten())[pts_add_map.squeeze(0).squeeze(-1)]] = True
                pts_add_map_total = pts_add_map_total.view(flow_mask_curr.shape[0], flow_mask_curr.shape[1])
        else:
            flow_gt = None
            indices_past_continue = None
            mask_past_die = None
            pts_add_map_total = None
            pts_curr = None
        

        self.gaussians_add(frame, dyna_mask, depth_highgrad_mask, indices_past_continue, mask_past_die, pts_add_map_total, pts_curr)
        
        if (self.time + 1) % self.gaussian_update_frame == 0 or self.time == 0:
            self.optimize_frames_ids.append(frame_id)
            is_keyframe = self.check_keyframe(frame, frame_id)
            move_to_gpu(frame)
            if self.dataset_type == "Scannetpp":
                self.local_optimize(frame, optimization_params)
                if is_keyframe:
                    self.global_optimization(
                        optimization_params,
                        select_keyframe_num=self.global_keyframe_num
                    )
            else:
                if not is_keyframe or self.get_stable_num <= 0:
                    self.local_optimize(frame, optimization_params, dyna_mask, depth_highgrad_mask)
                else:
                    self.global_optimization(
                        optimization_params,
                        select_keyframe_num=self.global_keyframe_num
                    )
                self.gaussians_delete(unstable=False)
        confidence = self.pointcloud.get_confidence
        print(
            "confidence:",
            "min", confidence.min().item(),
            "mean", confidence.float().mean().item(),
            "max", confidence.max().item(),
            "above threshold",
            (confidence > self.stable_confidence_thres).sum().item(),
        )
        self.gaussians_fix()
        self.error_gaussians_remove()
        self.gaussians_delete()
        
        if self.time>0:
            with torch.no_grad():
                if self.dyna_sample_mask is not None:
                    if pts_warp.shape[0]>0:
                        #THis i the main func of where the dynamic gaussian get the tangent and predict the future position of the dynamic gaussian
                        self.interp_extrap(indices_past_continue_fine.squeeze(0).squeeze(-1), mask_past_die, pts_add_map.squeeze(0).squeeze(-1), pts_curr, pts_past, pts_warp, t_past, t_curr, t_pred, dyna_mask_eval)
                    #   pass #comment out for interp_extrap
        move_to_cpu(frame)
        
    #Only chnages i made are removing the .cpu() as they where givning erros
    def interp_extrap(self, indices_past_continue_fine, mask_past_die, pts_add_map, pts_curr, pts_past, pts_warp, t_past, t_curr, t_pred, dyna_mask_eval):
        t_norm = (t_pred-t_past)/(t_curr-t_past)
        
        # match past and current.
        # For interp, two ways, 1. 
        # For extrap, always use curr GS to predict the future. Use Pt' (warped Pt to Pt-1) to find the closest Pt. The speed St is obtained in two ways: 1. directly reverse the estimated scene flow from t->t-1 to t-1->t 2.estimate scene flow t-1->t from network again.  
        if self.time > 1:
            
            #estimate_tangent
            end_coord = self.dyna_pointcloud._xyz[:,0,:].detach().clone()
            start_coord = end_coord.detach().clone()
            
    ##      if pts_add_map.shape[0] > start_coord.shape[0]:
    ##          pts_warp = pts_warp[self.select_mask_true_sample_idx]
    ##          pts_add_map = pts_add_map[self.select_mask_true_sample_idx]
    ##      start_coord[self.past_new_border_idx:] = pts_warp[pts_add_map]
            # First enter the "new points" index space.
            new_pts_warp = pts_warp[pts_add_map]
            # These indices are relative to the list of new points.
            selected_new_idx = self.select_mask_true_sample_idx.to(
                device=new_pts_warp.device,
                dtype=torch.long,
            )
            sampled_new_pts_warp = new_pts_warp[selected_new_idx]
            # Number of new Gaussians actually added by sample_pixels().
            new_gaussian_num = start_coord.shape[0] - self.past_new_border_idx
            start_coord[self.past_new_border_idx:] = sampled_new_pts_warp

            start_coord[:self.past_new_border_idx] = pts_past[~mask_past_die]
            self.dyna_pointcloud_future.copy(self.dyna_pointcloud)
            self.dyna_pointcloud_future.detach()

            tangent_past = end_coord - start_coord
            tangent_curr = tangent_past.clone()
            
            tangent_past_continue = tangent_past[:self.past_new_border_idx]
            tangent_past_past_continue = self.tangent_past_past[:,0,:][~mask_past_die]
            #constant acceleration
            tangent_curr_continue = (tangent_past_continue-tangent_past_past_continue)/(t_past-self.t_past_past)*(t_curr-t_past)+tangent_past_continue
            
            
            self.dyna_pointcloud_future._xyz = self.cubic_hermite(start_coord, tangent_past, end_coord, tangent_curr, t_norm)
            self.dyna_pointcloud_future._xyz = self.dyna_pointcloud_future._xyz[:, None, :].repeat(1,5,1)
            
            
            
            #tangent_curr = 2*tangent_past - self.tangent_past_past #naive version assuming equal time invervals 
            #tangent_curr = (tangent_past-self.tangent_past_past)/(t_past-self.t_past_past)*(t_curr-self.t_past_past)+self.tangent_past_past
            #tangent_curr = (tangent_past-self.tangent_past_past)/(t_past-self.t_past_past)*(t_curr-t_past)+tangent_past
            self.tangent_past_past = tangent_past[:, None, :].repeat(1,5,1)
            
            self.t_past_past = t_past    
            opt_frame = self.processed_frames[-1]
            render_pred = self.renderer.render(
                    #opt_frame, #ablation on rendering to curr frame instead of the target frame
                    self.frame_eval,
                    self.global_params_future,
                    #self.global_params, #ablation on not using motion function to move dyna GS and directly render in curr frame 
                )
            pred_rgb = render_pred['render'] #self.processed_frames[-1].original_image #render_pred['render']
            
            root_save_dir = os.path.join(self.save_path, "eval_pred", "frame_%04d"%(self.time))
            if not os.path.exists(root_save_dir):
                os.makedirs(root_save_dir)
            gt_image_pred = self.frame_eval.original_image
            masked_gt_image_pred_dyna = gt_image_pred.permute(1,2,0).cpu() + 0.3 * dyna_mask_eval[:,:,None]
            #plt.imshow(masked_gt_image_pred_dyna)
            #plt.axis("off")
            #plt.show()
            torchvision.utils.save_image(masked_gt_image_pred_dyna.permute(2,0,1),os.path.join(root_save_dir, "gt.png"))
                
            masked_pred_rgb_dyna = pred_rgb.permute(1,2,0).detach().cpu() + 0.3*dyna_mask_eval[:,:,None]
            #plt.imshow(masked_pred_rgb_dyna)
            #plt.axis("off")
            #plt.show()
            torchvision.utils.save_image(masked_pred_rgb_dyna.permute(2,0,1),os.path.join(root_save_dir, "pred.png"))
            
            masked_original_image_dyna = self.processed_frames[-2].original_image.permute(1,2,0).detach().cpu() + 0.3*dyna_mask_eval[:,:,None]
            #plt.imshow(masked_original_image_dyna)
            #plt.axis("off")
            #plt.show()
            torchvision.utils.save_image(masked_original_image_dyna.permute(2,0,1),os.path.join(root_save_dir, "t0.png"))
            masked_original_image_dyna = self.processed_frames[-1].original_image.permute(1,2,0).detach().cpu() + 0.3*dyna_mask_eval[:,:,None]
            #plt.imshow(masked_original_image_dyna)
            #plt.axis("off")
            #plt.show()
            torchvision.utils.save_image(masked_original_image_dyna.permute(2,0,1),os.path.join(root_save_dir, "t1.png"))

            psnr_value = psnr(gt_image_pred, pred_rgb.detach()).mean()
            self.psnr_pred.append(psnr_value)
            print("psnr_pred = ", psnr_value)
            if np.sum(dyna_mask_eval) > 0:
                gt_image_pred_dyna = (gt_image_pred.view(3,-1)[:, dyna_mask_eval.flatten()]).view(3,-1)
                pred_rgb_dyna = (pred_rgb.view(3,-1)[:, dyna_mask_eval.flatten()]).view(3,-1)
                psnr_value_dyna = psnr(gt_image_pred_dyna, pred_rgb_dyna.detach()).mean()
                self.psnr_pred_dyna.append(psnr_value_dyna)
                print("psnr_pred_dyna = ", psnr_value_dyna)
            
            
            image_error = (gt_image_pred - pred_rgb.detach()).abs()
            
            '''
            # Create a figure with 1 row and 3 columns
            fig, axes = plt.subplots(2, 3, figsize=(12, 4))  # 2 row, 3 columns

            # Display images in each subplot
            axes[0,0].imshow(pred_rgb.permute(1,2,0).detach().cpu().numpy())
            axes[0,0].set_title("pred_interp")
            axes[0,0].axis("off")  # Hide axes

            axes[0,1].imshow(gt_image_pred.permute(1,2,0).detach().cpu().numpy())
            axes[0,1].set_title("gt_interp")
            axes[0,1].axis("off")
            
            axes[0,2].imshow(image_error.permute(1,2,0).detach().cpu().numpy())
            axes[0,2].set_title("error_interp")
            axes[0,2].axis("off")
            
            axes[1,0].imshow(self.processed_frames[-2].original_image.permute(1,2,0).detach().cpu().numpy())
            axes[1,0].set_title("gt_start")
            axes[1,0].axis("off")
            
            axes[1,1].imshow(self.processed_frames[-1].original_image.permute(1,2,0).detach().cpu().numpy())
            axes[1,1].set_title("gt_end")
            axes[1,1].axis("off")
            
            diff_gt_start_interp = (gt_image_pred - self.processed_frames[-2].original_image).abs()
            axes[1,2].imshow(diff_gt_start_interp.permute(1,2,0).detach().cpu().numpy())
            axes[1,2].set_title("diff_gt_start_interp")
            axes[1,2].axis("off")

            # Show the plot
            plt.show()
            '''

        else:
            #linear_interp
            #get start and end for continuous
            
            end_coord = self.dyna_pointcloud._xyz[:,0,:].detach().clone()
            start_coord = end_coord.clone().detach().clone()
            
            ##      if pts_add_map.shape[0] > start_coord.shape[0]:
    ##          pts_warp = pts_warp[self.select_mask_true_sample_idx]
    ##          pts_add_map = pts_add_map[self.select_mask_true_sample_idx]
    ##      start_coord[self.past_new_border_idx:] = pts_warp[pts_add_map]
            # First enter the "new points" index space.
            new_pts_warp = pts_warp[pts_add_map]
            # These indices are relative to the list of new points.
            selected_new_idx = self.select_mask_true_sample_idx.to(
                device=new_pts_warp.device,
                dtype=torch.long,
            )
            sampled_new_pts_warp = new_pts_warp[selected_new_idx]
            # Number of new Gaussians actually added by sample_pixels().
            new_gaussian_num = start_coord.shape[0] - self.past_new_border_idx
            start_coord[self.past_new_border_idx:] = sampled_new_pts_warp

            start_coord[:self.past_new_border_idx] = pts_past[~mask_past_die]
            start_coord = start_coord[:, None, :].repeat(1,5,1)
            end_coord = end_coord[:, None, :].repeat(1,5,1)
            
            #splat to new-add
            # predict the future 
            self.dyna_pointcloud_future.copy(self.dyna_pointcloud)
            self.dyna_pointcloud_future.detach()
            self.dyna_pointcloud_future._xyz, self.tangent_past_past = self.linear_predict(start_coord, end_coord, t_norm)
            self.t_past_past = t_past
            
            
            opt_frame = self.processed_frames[-1]
            render_pred = self.renderer.render(
                    #opt_frame, #ablation on rendering to curr frame instead of the target frame
                    self.frame_eval,
                    self.global_params_future,
                    #self.global_params, #ablation on not using motion function to move dyna GS and directly render in curr frame 
                )
            pred_rgb = render_pred['render'] #self.processed_frames[-1].original_image #render_pred['render']
            
            gt_image_pred = self.frame_eval.original_image
            gt_image_pred_dyna = (gt_image_pred.view(3,-1)[:, dyna_mask_eval.flatten()]).view(3,-1)
            pred_rgb_dyna = (pred_rgb.view(3,-1)[:, dyna_mask_eval.flatten()]).view(3,-1)
            psnr_value = psnr(gt_image_pred, pred_rgb.detach()).mean()
            self.psnr_pred.append(psnr_value)
            psnr_value_dyna = psnr(gt_image_pred_dyna, pred_rgb_dyna.detach()).mean()
            self.psnr_pred_dyna.append(psnr_value_dyna)
            
            
            '''
            # Create a figure with 1 row and 3 columns
            fig, axes = plt.subplots(2, 3, figsize=(12, 4))  # 2 row, 3 columns

            # Display images in each subplot
            axes[0,0].imshow(pred_rgb.permute(1,2,0).detach().cpu().numpy())
            axes[0,0].set_title("pred_interp")
            axes[0,0].axis("off")  # Hide axes

            axes[0,1].imshow(gt_image_pred.permute(1,2,0).detach().cpu().numpy())
            axes[0,1].set_title("gt_interp")
            axes[0,1].axis("off")
            
            axes[0,2].imshow(image_error.permute(1,2,0).detach().cpu().numpy())
            axes[0,2].set_title("error_interp")
            axes[0,2].axis("off")
            
            axes[1,0].imshow(self.processed_frames[-2].original_image.permute(1,2,0).detach().cpu().numpy())
            axes[1,0].set_title("gt_start")
            axes[1,0].axis("off")
            
            axes[1,1].imshow(self.processed_frames[-1].original_image.permute(1,2,0).detach().cpu().numpy())
            axes[1,1].set_title("gt_end")
            axes[1,1].axis("off")
            
            diff_gt_start_interp = (gt_image_pred - self.processed_frames[-2].original_image).abs()
            axes[1,2].imshow(diff_gt_start_interp.permute(1,2,0).detach().cpu().numpy())
            axes[1,2].set_title("diff_gt_start_interp")
            axes[1,2].axis("off")

            # Show the plot
            plt.show()'''
            

        
        print("psnr_pred_mean = ", sum(self.psnr_pred)/len(self.psnr_pred))
        print("psnr_pred_dyna_mean = ", sum(self.psnr_pred_dyna)/len(self.psnr_pred_dyna))
        
    
    def cubic_hermite(self, start_coord, tangent_past, end_coord, tangent_curr, t):        
        return (2*t*(t**2)-3*(t**2)+1)*start_coord+(t*(t**2)-2*(t**2)+t)*tangent_past+(-2*t*(t**2)+3*(t**2))*end_coord+(t*(t**2)-(t**2))*tangent_curr
        
    
    def linear_predict(self, start, end, t_norm):
        tangent = end - start
        return end + (t_norm-1) * tangent, tangent
        
    #As in the legacy mmaper.py file in the gaussian add there are thse filter functions filter + attach, but where removed as i added them back as it was naturall thing to do
    def gaussians_add(self, frame, dyna_mask, depth_highgrad_mask, indices_past_continue, mask_past_die, pts_add_map, pts_curr):
        self.temp_points_init(frame, dyna_mask, depth_highgrad_mask)
        self.temp_points_filter()
        self.temp_points_attach(frame)
        self.temp_to_optimize()
        if not torch.sum(dyna_mask.to(torch.int64))==0:
            self.dyna_points_add(dyna_mask, depth_highgrad_mask, indices_past_continue, mask_past_die, pts_add_map, pts_curr)

    def update_poses(self, new_poses):
        if new_poses is None:
            return
        for frame in self.processed_frames:
            frame.updatePose(new_poses[frame.uid])

        for frame in self.keyframe_list:
            frame.updatePose(new_poses[frame.uid])

    def local_optimize(self, frame, update_args, dyna_mask, depth_highgrad_mask):
        print("===== map optimize =====")
        l = self.pointcloud.parametrize(update_args)
        history_stat = {
            "opacity": self.pointcloud._opacity.detach().clone(),
            "confidence": self.pointcloud.get_confidence.detach().clone(),
            "xyz": self.pointcloud._xyz.detach().clone(),
            "features_dc": self.pointcloud._features_dc.detach().clone(),
            "features_rest": self.pointcloud._features_rest.detach().clone(),
            "scaling": self.pointcloud._scaling.detach().clone(),
            "rotation": self.pointcloud.get_rotation.detach().clone(),
            "rotation_raw": self.pointcloud._rotation.detach().clone(),
        }
        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        # Previously the xyz componenet of l_dyna was not there, but as this is local optim, and we do want both static and dynamic pointcloud to be optimised togeteher, so i have added it
        l_dyna = self.dyna_pointcloud.parametrize_dyna(update_args)
        history_stat_dyna = {
            "opacity": self.dyna_pointcloud._opacity.detach().clone(),
            "confidence": self.dyna_pointcloud.get_confidence.detach().clone(),
            "xyz": self.dyna_pointcloud._xyz.detach().clone(),
            "features_dc": self.dyna_pointcloud._features_dc.detach().clone(),
            "features_rest": self.dyna_pointcloud._features_rest.detach().clone(),
            "scaling": self.dyna_pointcloud._scaling.detach().clone(),
            "rotation": self.dyna_pointcloud.get_rotation.detach().clone(),
            "rotation_raw": self.dyna_pointcloud._rotation.detach().clone(),
        }
        self.dyna_optimizer = torch.optim.Adam(l_dyna, lr=0.0, eps=1e-15)
        
        gaussian_update_iter = self.gaussian_update_iter
        self.render_masks = []
        self.tile_masks = []
        for frame in self.processed_frames:
            render_mask, tile_mask, render_ratio = self.evaluate_render_range(frame)
            self.render_masks.append(render_mask)
            self.tile_masks.append(tile_mask)
            if self.verbose:
                tile_raito = 1
                if tile_mask is not None:
                    tile_raito = tile_mask.sum() / torch.numel(tile_mask)
                print("tile mask ratio: {:f}".format(tile_raito))
        print(
            "unstable gaussian num = {:d}, stable gaussian num = {:d}, dyna gaussian num = {:d}".format(
                self.get_unstable_num, self.get_stable_num, self.get_dyna_num
            )
        )
        self.dyna_gaussian_num.append(self.get_dyna_num)

        with tqdm(total=gaussian_update_iter, desc="map update") as pbar:
            for iter in range(gaussian_update_iter):
                self.iter = iter
                random_index = random.randint(0, len(self.processed_frames) - 1)
                if iter > gaussian_update_iter / 2:
                    random_index = -1
              ##random_index = len(self.processed_frames) - 1    #This is also00 percent a mistake and so commented out
                opt_frame = self.processed_frames[random_index]
                opt_frame_map = self.processed_map[random_index]
                opt_render_mask = self.render_masks[random_index]
                opt_tile_mask = self.tile_masks[random_index]
                # compute loss
                render_ouput = self.renderer.render(
                    opt_frame,
                    self.global_params,
                )
                
                image_input = {
                    "color_map": devF(opt_frame_map["color_map"]),
                    "depth_map": devF(opt_frame_map["depth_map"]),
                    "normal_map": devF(opt_frame_map["normal_map_w"]),
                }
                
                loss, reported_losses = self.loss_update(
                    render_ouput,
                    image_input,
                    history_stat,
                    history_stat_dyna,
                    update_args, dyna_mask, depth_highgrad_mask,
                    render_mask=opt_render_mask,
                    unstable=True,
                )
                pbar.set_postfix({"loss": "{0:1.5f}".format(loss)})
                pbar.update(1)

                
        self.pointcloud.detach()
        self.iter = 0

    # Fix the confidence points
    def gaussians_fix(self, mask=None):
        if mask is None:
            confidence_mask = (
                self.pointcloud.get_confidence > self.stable_confidence_thres
            ).squeeze()
            stable_mask = confidence_mask
        else:
            stable_mask = mask.squeeze()
        if self.verbose:
            print("===== points fix =====")
            print(
                "fix gaussian num: {:d}".format(stable_mask.sum()),
            )
        if stable_mask.sum() > 0:
            stable_params = self.pointcloud.remove(stable_mask)
            stable_params["confidence"] = torch.clip(
                stable_params["confidence"], max=self.stable_confidence_thres
            )
            self.stable_pointcloud.cat(stable_params)

    # Fix the confidence points
    def gaussians_release(self, mask):
        if mask.sum() > 0:
            unstable_params = self.stable_pointcloud.remove(mask)
            unstable_params["confidence"] = devF(
                torch.zeros_like(unstable_params["confidence"])
            )
            unstable_params["add_tick"] = self.time * devF(
                torch.ones_like(unstable_params["add_tick"])
            )
            unstable_params["color_error_counter"] = devI(
            torch.zeros_like(unstable_params["color_error_counter"])
            )
            unstable_params["depth_error_counter"] = devI(
            torch.zeros_like(unstable_params["depth_error_counter"])
         )
         ## self.stable_pointcloud.cat(unstable_params)
            self.pointcloud.cat(unstable_params)

    # Remove too small/big gaussians, long time unstable gaussians, insolated_gaussians
    def gaussians_delete(self, unstable=True):
        if unstable:
            pointcloud = self.pointcloud
        else:
            pointcloud = self.stable_pointcloud
        if pointcloud.get_points_num == 0:
            return
        threshold = self.KNN_threshold
        big_gaussian_mask = (
            pointcloud.get_radius > (pointcloud.get_radius.mean() * 10)
        ).squeeze()
        unstable_time_mask = (
            (self.time - pointcloud.get_add_tick) > self.unstable_time_window
        ).squeeze()
        if unstable:
            delete_mask = (
                big_gaussian_mask | unstable_time_mask 
            )
        else:
            delete_mask = big_gaussian_mask 
        if self.verbose:
            print("===== points delete =====")
            print(
                "threshold: {:.1f} cm, big num: {:d}, unstable num: {:d}, delete num: {:d}".format(
                    threshold * 100,
                    big_gaussian_mask.sum(),
                    unstable_time_mask.sum(),
                    delete_mask.sum(),
                ),
            )
        pointcloud.delete(delete_mask)

    # check if current frame is a keyframe
    def check_keyframe(self, frame, frame_id):
        # add keyframe
        if self.time == 0:
            self.keyframe_list.append(frame.move_to_cpu_clone())
            self.keyframe_ids.append(frame_id)
            image_input = {
                "color_map": self.frame_map["color_map"].detach().cpu(),
                "depth_map": self.frame_map["depth_map"].detach().cpu(),
                "normal_map": self.frame_map["normal_map_w"].detach().cpu(),
            }
            self.keymap_list.append(image_input)
            return False
        prev_rot = self.keyframe_list[-1].R.T
        prev_trans = self.keyframe_list[-1].T
        curr_rot = frame.R.T
        curr_trans = frame.T
        _, theta_diff = rot_compare(prev_rot, curr_rot)
        _, l2_diff = trans_compare(prev_trans, curr_trans)
        frame_gap = frame_id - self.keyframe_ids[-1]
        if self.verbose:
            print("rot diff: {:.2f}, move diff: {:.2f}".format(theta_diff, l2_diff))
        if theta_diff > self.keyframe_theta_thes or l2_diff > self.keyframe_trans_thes or frame_gap >= 40:
            print("add key frame at frame {:d}!".format(self.time))
            image_input = {
                "color_map": self.frame_map["color_map"].detach().cpu(),
                "depth_map": self.frame_map["depth_map"].detach().cpu(),
                "normal_map": self.frame_map["normal_map_w"].detach().cpu(),
            }
            self.keyframe_list.append(frame.move_to_cpu_clone())
            self.keymap_list.append(image_input)
            self.keyframe_ids.append(frame_id)
            return True
        else:
            return False
    #Made the dyna_mask and depth_highgrad_mask optional as they are not used in the loss_update function. 
    # update confidence by grad
    def loss_update(
        self,
        render_output,
        image_input,
        init_stat,
        init_stat_dyna,
        update_args,
        dyna_mask = None,
        depth_highgrad_mask = None,
        render_mask=None,
        unstable=True,
    ):
        if unstable:
            pointcloud = self.pointcloud
        else:
            pointcloud = self.stable_pointcloud
        opacity = pointcloud.opacity_activation(init_stat["opacity"])
        attach_mask = (opacity < 0.9).squeeze()
        attach_loss = torch.tensor(0)
        if attach_mask.sum() > 0:
            attach_loss = 1000 * (
                l2_loss(
                    pointcloud._scaling[attach_mask],
                    init_stat["scaling"][attach_mask],
                )
                + l2_loss(
                    pointcloud._xyz[attach_mask],
                    init_stat["xyz"][attach_mask],
                )
                + l2_loss(
                    pointcloud._rotation[attach_mask],
                    init_stat["rotation_raw"][attach_mask],
                )
            )
        # ALso 100 percent a mistake as, dyna_pointlcoud does not have any use for increasin in confidence whihc it was being used below, line 819
        dyna_pointcloud = self.dyna_pointcloud
        opacity = dyna_pointcloud.opacity_activation(init_stat_dyna["opacity"])
        attach_mask = (opacity < 0.9).squeeze()
        attach_loss_dyna = torch.tensor(0)
        if attach_mask.sum() > 0 and "xyz" in init_stat_dyna:
            attach_loss_dyna = 1000 * (
                l2_loss(
                    dyna_pointcloud._scaling[attach_mask],
                    init_stat_dyna["scaling"][attach_mask],
                )
                + l2_loss(
                    dyna_pointcloud._xyz[attach_mask],
                    init_stat_dyna["xyz"][attach_mask],
                )
                + l2_loss(
                    dyna_pointcloud._rotation[attach_mask],
                    init_stat_dyna["rotation_raw"][attach_mask],
                )
            )
        
        image, depth, normal, depth_index = (
            render_output["render"].permute(1, 2, 0),
            render_output["depth"].permute(1, 2, 0),
            render_output["normal"].permute(1, 2, 0),
            render_output["depth_index_map"].permute(1, 2, 0),
        )
        ssim_loss = devF(torch.tensor(0))
        normal_loss = devF(torch.tensor(0))
        depth_loss = devF(torch.tensor(0))
        if render_mask is None:
            render_mask = devB(torch.ones(image.shape[:2]))
            ssim_loss = 1 - ssim(image.permute(2,0,1), image_input["color_map"].permute(2,0,1))
        else:
            render_mask = render_mask.bool()
        if self.dataset_type == "Scannetpp":
            render_mask = render_mask & (image_input["depth_map"] > 0).squeeze()
        img_mask = render_mask
        color_loss = l1_loss(image, image_input["color_map"])

        if depth is not None and update_args.depth_weight > 0:
            depth_error = depth - image_input["depth_map"]
            valid_depth_mask = (
                (depth_index != -1).squeeze()
                & (image_input["depth_map"] > 0).squeeze()
                & (depth_error < self.add_depth_thres).squeeze()
            )
            depth_loss = torch.abs(depth_error[valid_depth_mask]).mean()

        if normal is not None and update_args.normal_weight > 0:
            cos_dist = 1 - F.cosine_similarity(
                normal, image_input["normal_map"], dim=-1
            )
            valid_normal_mask = (
                render_mask
                & (depth_index != -1).squeeze()
                & (~(image_input["normal_map"] == 0).all(dim=-1))
            )
            normal_loss = cos_dist[valid_normal_mask].mean()
            
            
        total_loss = (
            update_args.depth_weight * depth_loss
            + update_args.normal_weight * normal_loss
            + update_args.color_weight * color_loss
            + update_args.ssim_weight * ssim_loss
        )
        loss = total_loss
        (loss + attach_loss + attach_loss_dyna).backward()
        
        self.optimizer.step()
        if "xyz" in init_stat_dyna:
            self.dyna_optimizer.step()

        # update confidence by grad
        if pointcloud._features_dc.grad is not None: # when no dyna GS, pointcloud._features_dc.grad is none
            grad_mask = (pointcloud._features_dc.grad.abs() != 0).any(dim=-1)
            pointcloud._confidence[grad_mask] += 1

        # report train loss
        report_losses = {
            "total_loss": total_loss.item(),
            "depth_loss": depth_loss.item(),
            "ssim_loss": ssim_loss.item(),
            "normal_loss": normal_loss.item(),
            "color_loss": color_loss.item(),
            "scale_loss": attach_loss.item(),
        }
        self.train_report(self.get_total_iter, report_losses)
        self.optimizer.zero_grad(set_to_none=True)
        if "xyz" in init_stat_dyna:
            self.dyna_optimizer.zero_grad(set_to_none=True)
        return loss, report_losses

    def evaluate_render_range(
        self, frame, global_opt=False, sample_ratio=-1, unstable=True
    ):
        if unstable:
            render_output = self.renderer.render(
                frame,
                self.unstable_params
            )
        else:
            render_output = self.renderer.render(
                frame,
                self.stable_params
            )
        unstable_T_map = render_output["T_map"]

        if global_opt:
            if sample_ratio > 0:
                render_image = render_output["render"].permute(1, 2, 0)
                gt_image = frame.original_image.permute(1, 2, 0).cuda()
                image_diff = (render_image - gt_image).abs()
                color_error = torch.sum(
                    image_diff, dim=-1, keepdim=False
                )
                filter_mask = (render_image.sum(dim=-1) == 0)
                color_error[filter_mask] = 0
                tile_mask = colorerror2tilemask(color_error, 16, sample_ratio)
                render_mask = (
                    F.interpolate(
                        tile_mask.float().unsqueeze(0).unsqueeze(0),
                        scale_factor=16,
                        mode="nearest",
                    )
                    .squeeze(0)
                    .squeeze(0)
                    .bool()
                )[: color_error.shape[0], : color_error.shape[1]]
            # after training, real global optimization
            else:
                render_mask = (unstable_T_map != 1).squeeze(0)
                tile_mask = None
        else:
            render_mask = (unstable_T_map != 1).squeeze(0)
            tile_mask = transmission2tilemask(render_mask, 16, 0.5)

        render_ratio = render_mask.sum() / self.get_pixel_num
        return render_mask, tile_mask, render_ratio

    def error_gaussians_remove(self):
        if self.get_stable_num <= 0:
            return
        # check error by backprojection
        check_frame = self.processed_frames[-1]
        check_map = self.processed_map[-1]
        render_output = self.renderer.render(
            check_frame, self.global_params
        )
        # [unstable, stable]
        unstable_points_num = self.get_unstable_num
        stable_points_num = self.get_stable_num

        color = render_output["render"].permute(1, 2, 0)
        depth = render_output["depth"].permute(1, 2, 0)
        normal = render_output["normal"].permute(1, 2, 0)
        depth_index = render_output["depth_index_map"].permute(1, 2, 0)
        color_index = render_output["color_index_map"].permute(1, 2, 0)

        depth_error = torch.abs(check_map["depth_map"] - depth)
        depth_error[(check_map["depth_map"] - depth) < 0] = 0
        image_error = torch.abs(check_map["color_map"] - color)
        color_error = torch.sum(image_error, dim=-1, keepdim=True)

        normal_error = devF(torch.zeros_like(depth_error))
        invalid_mask = (check_map["depth_map"] == 0) | (depth_index == -1)
        invalid_mask = invalid_mask.squeeze()

        depth_error[invalid_mask] = 0
        color_error[check_map["depth_map"] == 0] = 0
        normal_error[invalid_mask] = 0
        H, W = self.frame_map["color_map"].shape[:2]
        P = unstable_points_num + stable_points_num
        (
            gaussian_color_error,
            gaussian_depth_error,
            gaussian_normal_error,
            outlier_count,
        ) = accumulate_gaussian_error(
            H,
            W,
            P,
            color_error,
            depth_error,
            normal_error,
            color_index,
            depth_index,
            self.add_color_thres,
            self.add_depth_thres,
            self.add_normal_thres,
            True,
        )

        color_filter_thres = 2 * self.add_color_thres
        depth_filter_thres = 2 * self.add_depth_thres

        depth_delete_mask = (gaussian_depth_error > depth_filter_thres).squeeze()
        color_release_mask = (gaussian_color_error > color_filter_thres).squeeze()
        if self.verbose:
            print("===== outlier remove =====")
            print(
                "color outlier num: {:d}, depth outlier num: {:d}".format(
                    (color_release_mask).sum(),
                    (depth_delete_mask).sum(),
                ),
            )

        depth_delete_mask_stable = depth_delete_mask[unstable_points_num:, ...]
        color_release_mask_stable = color_release_mask[unstable_points_num:, ...]

        self.stable_pointcloud._depth_error_counter[depth_delete_mask_stable] += 1
        self.stable_pointcloud._color_error_counter[color_release_mask_stable] += 1

        delete_thresh = 10
        depth_delete_mask = (
            self.stable_pointcloud._depth_error_counter >= delete_thresh
        ).squeeze()
        color_release_mask = (
            self.stable_pointcloud._color_error_counter >= delete_thresh
        ).squeeze()
        self.stable_pointcloud.delete(depth_delete_mask)
        self.gaussians_release(color_release_mask[~depth_delete_mask])

    # update all stable gaussians by keyframes render
    def global_optimization(
        self, update_args, select_keyframe_num=-1, is_end=False
    ):
        print("===== global optimize =====")
        if select_keyframe_num == -1:
            self.gaussians_fix(mask=(self.pointcloud.get_confidence > -1))
        print(
            "keyframe num = {:d}, stable gaussian num = {:d}".format(
                self.get_keyframe_num, self.get_stable_num
            )
        )
        if self.get_stable_num == 0:
            return
        l = self.stable_pointcloud.parametrize(update_args)
        if select_keyframe_num != -1:
            l[0]["lr"] = 0
            for i in range(1, len(l)):
                l[i]["lr"] *= 0.1
        else:
            l[0]["lr"] = 0.0000
            l[1]["lr"] *= self.feature_lr_coef
            l[2]["lr"] *= self.feature_lr_coef
            l[4]["lr"] *= self.scaling_lr_coef
            l[5]["lr"] *= self.rotation_lr_coef
        is_final = False
        #Again here even though thge loss update need this info below, it was not computed but need to be, its xyz are comnmented out such that no loss is computed for the dynamic
        history_stat_dyna = {
            "opacity": self.dyna_pointcloud._opacity.detach().clone(),
            "confidence": self.dyna_pointcloud.get_confidence.detach().clone(),
 ##         "xyz": self.dyna_pointcloud._xyz.detach().clone(),
            "features_dc": self.dyna_pointcloud._features_dc.detach().clone(),
            "features_rest": self.dyna_pointcloud._features_rest.detach().clone(),
            "scaling": self.dyna_pointcloud._scaling.detach().clone(),
            "rotation": self.dyna_pointcloud.get_rotation.detach().clone(),
            "rotation_raw": self.dyna_pointcloud._rotation.detach().clone(),
        }
        init_stat = {
            "opacity": self.stable_pointcloud._opacity.detach().clone(),
            "scaling": self.stable_pointcloud._scaling.detach().clone(),
            "xyz": self.stable_pointcloud._xyz.detach().clone(),
            "rotation_raw": self.stable_pointcloud._rotation.detach().clone(),
        }
        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        total_iter = int(self.gaussian_update_iter)
        sample_ratio = 0.4
        if select_keyframe_num == -1:
            total_iter = self.get_keyframe_num * self.final_global_iter
            is_final = True
            select_keyframe_num = self.get_keyframe_num
            update_args.depth_weight = 0
            sample_ratio = -1

        # test random kframes
        random_kframes = False
        
        select_keyframe_num = min(select_keyframe_num, self.get_keyframe_num)
        if random_kframes:
            if select_keyframe_num >= self.get_keyframe_num:
                select_kframe_indexs = list(range(0, self.get_keyframe_num))
            else:
                select_kframe_indexs = np.random.choice(np.arange(1, min(select_keyframe_num * 2, self.get_keyframe_num)),
                                                        select_keyframe_num-1,
                                                        replace=False).tolist() + [0]
        else:
            select_kframe_indexs = list(range(select_keyframe_num))
        
        select_kframe_indexs = [i*-1-1 for i in select_kframe_indexs]
            
            
        select_frame = []
        select_map = []
        select_render_mask = []
        select_tile_mask = []
        select_id = []
        # TODO: only sample some pixels of keyframe for global optimization
        for index in select_kframe_indexs:
            move_to_gpu(self.keyframe_list[index])
            move_to_gpu_map(self.keymap_list[index])
            select_frame.append(self.keyframe_list[index])
            select_map.append(self.keymap_list[index])
            render_mask, tile_mask, _ = self.evaluate_render_range(
                self.keyframe_list[index],
                global_opt=True,
                unstable=False,
                sample_ratio=sample_ratio,
            )
            if select_keyframe_num == -1:
                move_to_cpu(self.keyframe_list[index])
                move_to_cpu_map(self.keymap_list[index])
            select_render_mask.append(render_mask)
            select_tile_mask.append(tile_mask)
            select_id.append(self.keyframe_ids[index])
            if self.verbose:
                tile_raito = 1
                if tile_mask is not None:
                    tile_raito = tile_mask.sum() / torch.numel(tile_mask)

        with tqdm(total=total_iter, desc="global optimization") as pbar:
            for iter in range(total_iter):
                self.iter = iter
                random_index = random.randint(0, select_keyframe_num - 1)
                frame_input = select_frame[random_index]
                image_input = select_map[random_index]
                if select_keyframe_num == -1:
                    move_to_gpu(frame_input)
                    move_to_gpu_map(image_input)
                if not random_kframes and iter > total_iter / 2 and not is_final:
                    random_index = -1
                render_ouput = self.renderer.render(
                    frame_input,
                    self.stable_params,
                    tile_mask=select_tile_mask[random_index],
                )
                loss, reported_losses = self.loss_update(
                    render_ouput,
                    image_input,
                    init_stat,
                    history_stat_dyna,
                    update_args,
                    render_mask=select_render_mask[random_index],
                    unstable=False,
                )
                if select_keyframe_num == -1:
                    move_to_cpu(frame_input)
                    move_to_cpu_map(image_input)
                pbar.set_postfix({"loss": "{0:1.5f}".format(loss)})
                pbar.update(1)

        for index in range(-select_keyframe_num, 0):
            move_to_cpu(self.keyframe_list[index])
            move_to_cpu_map(self.keymap_list[index])
        self.stable_pointcloud.detach()

    def dyna_points_add(self, dyna_mask, depth_highgrad_mask, indices_past_continue, mask_past_die, pts_add_map, pts_curr):
        if self.time == 0:
            mask = dyna_mask #| depth_highgrad_mask
            xyz, normal, color, self.dyna_sample_mask, self.select_mask_true_sample_idx = sample_pixels(
                self.frame_map["vertex_map_w"],
                self.frame_map["normal_map_w"],
                self.frame_map["color_map"],
                self.uniform_sample_num,
                mask,
                static=False
            )

            object_id = self.frame_map["object_id_map"].to(xyz.device)[
                self.dyna_sample_mask
            ]

            self.dyna_pointcloud.add_empty_points(xyz, normal, color, self.time, object_id=object_id)
            
            self.dyna_pointcloud.update_geometry(torch.tensor([]).cuda(),
            torch.tensor([]).cuda(),
            )
        else:
            # first use "indices_past_continue" to update past GS
            dyna_pts_before = self.dyna_pointcloud._xyz[indices_past_continue.squeeze(0).squeeze(-1)]
            pts_curr_temp = torch.cat((pts_curr.unsqueeze(1), torch.zeros((pts_curr.shape[0], 4, pts_curr.shape[1])).cuda()), dim = 1)
            self.dyna_pointcloud._xyz = self.dyna_pointcloud._xyz.scatter(
    0, indices_past_continue.squeeze(0).squeeze(-1).unsqueeze(-1).expand(-1, 3).unsqueeze(1).expand(-1,5,-1), pts_curr_temp
)

            # use mask_past_die to remove outlier past GS
            self.dyna_pointcloud.delete(mask_past_die)
            
            # before the idx all past-continous, starting (after) from the idx all new added
            self.past_new_border_idx = self.dyna_pointcloud._xyz.shape[0]
            
            # use pts_add_map to add new curr GS 
            xyz, normal, color, self.dyna_sample_mask, self.select_mask_true_sample_idx = sample_pixels(
                self.frame_map["vertex_map_w"],
                self.frame_map["normal_map_w"],
                self.frame_map["color_map"],
                self.uniform_sample_num,
                pts_add_map,
                static=False
            )
            object_id = self.frame_map["object_id_map"].to(xyz.device)[
                self.dyna_sample_mask
            ]
            self.dyna_pointcloud.add_empty_points(xyz, normal, color, self.time, object_id=object_id)
            self.dyna_pointcloud.update_geometry(torch.tensor([]).cuda(),
            torch.tensor([]).cuda(),
            )

    # Sample some pixels as the init gaussians
    def temp_points_init(self, frame: Camera, dyna_mask, depth_highgrad_mask):
        # print("===== temp points add =====")
        if self.time == 0:
            depth_range_mask = torch.ones_like(self.frame_map["depth_map"]).to(bool)
            #As you can tell the code below totally overites the code above, but i think it is necessay as the map map filter unnecessay points and was being used in the legacy mapper.py
            depth_range_mask = (self.frame_map["depth_map"] > 0) & (~(dyna_mask | depth_highgrad_mask).unsqueeze(-1))
            xyz, normal, color, sampled_mask, _ = sample_pixels(
                self.frame_map["vertex_map_w"],
                self.frame_map["normal_map_w"],
                self.frame_map["color_map"],
                self.uniform_sample_num,
                depth_range_mask,
            )
            object_id = self.frame_map["object_id_map"].to(xyz.device)[sampled_mask]

            self.temp_pointcloud.add_empty_points(
                xyz, normal, color, self.time, object_id=object_id
            )
        else:
            self.get_render_output(frame)
            
            transmission_sample_mask = (
                self.model_map["render_transmission"] > self.add_transmission_thres
            )
            transmission_sample_ratio = (
                transmission_sample_mask.sum() / (self.get_pixel_num)
            )
            transmission_sample_num = devI(
                self.transmission_sample_ratio
                * transmission_sample_ratio
                * self.uniform_sample_num
            )

            if self.verbose:
                print(
                    "transmission empty num = {:d}, sample num = {:d}".format(
                        transmission_sample_mask.sum(), transmission_sample_num
                    )
                )
            if transmission_sample_num > 0:
                transmission_sample_mask = transmission_sample_mask & (~(dyna_mask | depth_highgrad_mask).unsqueeze(-1))
                xyz_trans, normal_trans, color_trans, sampled_mask, _ = sample_pixels(
                    self.frame_map["vertex_map_w"],
                    self.frame_map["normal_map_w"],
                    self.frame_map["color_map"],
                    transmission_sample_num,
                    transmission_sample_mask,
                )
                #Here even though they computed the xyz_trans and such but they never added it to the temp_pointcloud, so i have added it below
                object_id = self.frame_map["object_id_map"].to(xyz_trans.device)[sampled_mask]

                self.temp_pointcloud.add_empty_points(
                    xyz_trans, normal_trans, color_trans, self.time, object_id=object_id
                )

            depth_error = torch.abs(
                self.frame_map["depth_map"] - self.model_map["render_depth"]
            )
            color_error = torch.abs(
                self.frame_map["color_map"] - self.model_map["render_color"]
            ).mean(dim=-1, keepdim=True)

            depth_sample_mask = (
                (depth_error > self.add_depth_thres)
                & (self.frame_map["depth_map"] > 0)
                & (self.model_map["render_depth_index"] > -1)
            )
            color_sample_mask = (
                (color_error > self.add_color_thres)
                & (self.frame_map["depth_map"] > 0)
                & (self.model_map["render_transmission"] < self.add_transmission_thres)
            )
            sample_mask = color_sample_mask | depth_sample_mask
            sample_mask = sample_mask & (~transmission_sample_mask)
            sample_mask = sample_mask & (~(dyna_mask).unsqueeze(-1))
            sample_num = devI(sample_mask.sum() * self.error_sample_ratio)
            if self.verbose:
                print(
                    "wrong depth num = {:d}, wrong color num = {:d}, sample num = {:d}".format(
                        depth_sample_mask.sum(),
                        color_sample_mask.sum(),
                        sample_num,
                    )
                )
            if sample_num > 0:
                xyz_error, normal_error, color_error, sampled_mask, _ = sample_pixels(
                self.frame_map["vertex_map_w"],
                self.frame_map["normal_map_w"],
                self.frame_map["color_map"],
                sample_num,
                sample_mask,
            )
                object_id = self.frame_map["object_id_map"].to(xyz_error.device)[sampled_mask]
                self.temp_pointcloud.add_empty_points(
                    xyz_error, normal_error, color_error, self.time, object_id=object_id
                )

    # Remove temp points that fall within the existing unstable Gaussian.
    def temp_points_filter(self, topk=3):
        if self.get_unstable_num > 0:
            temp_xyz = self.temp_pointcloud.get_xyz
            if self.verbose:
                print("init {} temp points".format(self.temp_pointcloud.get_points_num))
            exist_xyz = self.unstable_params["xyz"]
            exist_raidus = self.unstable_params["radius"]
            if torch.numel(exist_xyz) > 0:
                inbbox_mask = bbox_filter(temp_xyz[:, 0, :], exist_xyz[:, 0, :])
                exist_xyz = exist_xyz[inbbox_mask]
                exist_raidus = exist_raidus[inbbox_mask]

            if torch.numel(exist_xyz) == 0:
                return

            nn_dist, nn_indices, _ = knn_points(
                temp_xyz[:, 0, :][None, ...],
                exist_xyz[:, 0, :][None, ...],
                norm=2,
                K=topk,
                return_nn=True,
            )
            nn_dist = torch.sqrt(nn_dist).squeeze(0)
            nn_indices = nn_indices.squeeze(0)

            corr_radius = exist_raidus[nn_indices] * 0.6
            inside_mask = (nn_dist < corr_radius).any(dim=-1)
            if self.verbose:
                print("delete {} temp points".format(inside_mask.sum().item()))
            self.temp_pointcloud.delete(inside_mask)

    # Attach gaussians fall with in the stable gaussians. attached gaussians is set to low opacity and fix scale
    def temp_points_attach(self, frame: Camera, unstable_opacity_low=0.1):
        if self.get_stable_num == 0:
            return
        # project unstable gaussians and compute uv
        unstable_xyz = self.temp_pointcloud.get_xyz
        origin_indices = torch.arange(unstable_xyz.shape[0]).cuda().long()
        unstable_opacity = self.temp_pointcloud.get_opacity
        unstable_opacity_filter = (unstable_opacity > unstable_opacity_low).squeeze(-1)
        unstable_xyz = unstable_xyz[unstable_opacity_filter]
        unstable_uv = frame.get_uv(unstable_xyz)
        indices = torch.arange(unstable_xyz.shape[0]).cuda().long()
        unstable_mask = (
            (unstable_uv[:, 0] >= 0)
            & (unstable_uv[:, 0] < frame.image_width)
            & (unstable_uv[:, 1] >= 0)
            & (unstable_uv[:, 1] < frame.image_height)
        )
        project_uv = unstable_uv[unstable_mask]

        # get the corresponding stable gaussians
        stable_render_output = self.renderer.render(
            frame, self.stable_params,
        )
        
        stable_index = stable_render_output["color_index_map"].permute(1, 2, 0)
        intersect_mask = stable_index[project_uv[:, 1], project_uv[:, 0]] >= 0
        indices = indices[unstable_mask][intersect_mask[:, 0]]

        # compute point to plane distance
        intersect_stable_index = (
            (stable_index[unstable_uv[indices, 1], unstable_uv[indices, 0]])
            .squeeze(-1)
            .long()
        )
        
        stable_normal_check = self.stable_pointcloud.get_normal[intersect_stable_index]
        stable_xyz_check = self.stable_pointcloud.get_xyz[intersect_stable_index]
        unstable_xyz_check = self.temp_pointcloud.get_xyz[indices]
        point_to_plane_distance = (
            (stable_xyz_check[:,0,:] - unstable_xyz_check[:,0,:]) * stable_normal_check
        ).sum(dim=-1)
        intersect_check = point_to_plane_distance.abs() < 0.5 * self.add_depth_thres
        indices = indices[intersect_check]
        indices = origin_indices[unstable_opacity_filter][indices]

        # set opacity
        self.temp_pointcloud._opacity[indices] = inverse_sigmoid(
            unstable_opacity_low
            * torch.ones_like(self.temp_pointcloud._opacity[indices])
        )
        if self.verbose:
            print("attach {} unstable gaussians".format(indices.shape[0]))

    # Initialize temp points as unstable gaussian.
    def temp_to_optimize(self):
        self.temp_pointcloud.update_geometry(
            self.global_params["xyz"],
            self.global_params["radius"],
        )
        if self.verbose:
            print("===== points add =====")
            print(
                "add new gaussian num: {:d}".format(self.temp_pointcloud.get_points_num)
            )
        remove_mask = devB(torch.ones(self.temp_pointcloud.get_points_num))
        temp_params = self.temp_pointcloud.remove(remove_mask)
        self.pointcloud.cat(temp_params)


    # detect isolated gaussians by KNN
    def gaussians_isolated(self, points, topk=5, threshold=0.005):
        if threshold < 0:
            isolated_mask = devB(torch.zeros(points.shape[0]))
            return isolated_mask
        nn_dist, nn_indices, _ = knn_points(
            points[None, ...],
            points[None, ...],
            norm=2,
            K=topk + 1,
            return_nn=True,
        )
        dist_mean = nn_dist[0, :, 1:].mean(1)
        isolated_mask = dist_mean > threshold
        return isolated_mask

    def create_workspace(self):
        if os.path.exists(self.save_path):
            shutil.rmtree(self.save_path)
        os.makedirs(self.save_path, exist_ok=True)
        render_save_path = os.path.join(self.save_path, "eval_render")
        os.makedirs(render_save_path, exist_ok=True)
        model_save_path = os.path.join(self.save_path, "save_model")
        os.makedirs(model_save_path, exist_ok=True)
        traj_save_path = os.path.join(self.save_path, "save_traj")
        os.makedirs(traj_save_path, exist_ok=True)
        traj_save_path = os.path.join(self.save_path, "eval_metric")
        os.makedirs(traj_save_path, exist_ok=True)

        if self.mode == "single process" and self.use_tensorboard:
            self.tb_writer = SummaryWriter(self.save_path)
        else:
            self.tb_writer = None

    def save_model(self, path=None, save_data=True, save_sibr=True, save_merge=True):
        if path == None:
            frame_name = "frame_{:04d}".format(self.time)
            model_save_path = os.path.join(self.save_path, "save_model", frame_name)
            os.makedirs(model_save_path, exist_ok=True)
            path = os.path.join(
                model_save_path,
                "iter_{:04d}".format(self.iter),
            )
        if save_data:
            self.pointcloud.save_model_ply(path + ".ply", include_confidence=True)
            self.stable_pointcloud.save_model_ply(
                path + "_stable.ply", include_confidence=True
            )
        if save_sibr:
            self.pointcloud.save_model_ply(path + "_sibr.ply", include_confidence=False)
            self.stable_pointcloud.save_model_ply(
                path + "_stable_sibr.ply", include_confidence=False
            )
        if self.get_unstable_num > 0 and self.get_stable_num > 0:
            if save_data and save_merge:
                merge_ply(
                    path + ".ply",
                    path + "_stable.ply",
                    path + "_merge.ply",
                    include_confidence=True,
                )
            if save_sibr and save_merge:
                merge_ply(
                    path + "_sibr.ply",
                    path + "_stable_sibr.ply",
                    path + "_merge_sibr.ply",
                    include_confidence=False,
                )

    def train_report(self, iteration, losses):
        if self.tb_writer is not None:
            for loss in losses:
                self.tb_writer.add_scalar(
                    "train/{}".format(loss), losses[loss], iteration
                )

    def eval_report(self, iteration, losses):
        if self.tb_writer is not None:
            for loss in losses:
                self.tb_writer.add_scalar(
                    "eval/{}".format(loss), losses[loss], iteration
                )

    def get_render_output(self, frame):
        render_output = self.renderer.render(
            frame, self.global_params,
        )
        self.model_map["render_color"] = render_output["render"].permute(1, 2, 0)
        self.model_map["render_depth"] = render_output["depth"].permute(1, 2, 0)
        self.model_map["render_normal"] = render_output["normal"].permute(1, 2, 0)
        self.model_map["render_color_index"] = render_output["color_index_map"].permute(
            1, 2, 0
        )
        self.model_map["render_depth_index"] = render_output["depth_index_map"].permute(
            1, 2, 0
        )
        self.model_map["render_transmission"] = render_output["T_map"].permute(1, 2, 0)

    @property
    def stable_params(self):
        have_stable = self.get_stable_num > 0
        xyz = self.stable_pointcloud.get_xyz if have_stable else torch.empty(0)
        opacity = self.stable_pointcloud.get_opacity if have_stable else torch.empty(0)
        scales = self.stable_pointcloud.get_scaling if have_stable else torch.empty(0)
        rotations = (
            self.stable_pointcloud.get_rotation if have_stable else torch.empty(0)
        )
        shs = self.stable_pointcloud.get_features if have_stable else torch.empty(0)
        radius = self.stable_pointcloud.get_radius if have_stable else torch.empty(0)
        normal = self.stable_pointcloud.get_normal if have_stable else torch.empty(0)
        confidence = (
            self.stable_pointcloud.get_confidence if have_stable else torch.empty(0)
        )
        stable_params = {
            "xyz": devF(xyz),
            "opacity": devF(opacity),
            "scales": devF(scales),
            "rotations": devF(rotations),
            "shs": devF(shs),
            "radius": devF(radius),
            "normal": devF(normal),
            "confidence": devF(confidence),
        }
        return stable_params

    @property
    def unstable_params(self):
        have_unstable = self.get_unstable_num > 0
        xyz = self.pointcloud.get_xyz if have_unstable else torch.empty(0)
        opacity = self.pointcloud.get_opacity if have_unstable else torch.empty(0)
        scales = self.pointcloud.get_scaling if have_unstable else torch.empty(0)
        rotations = self.pointcloud.get_rotation if have_unstable else torch.empty(0)
        shs = self.pointcloud.get_features if have_unstable else torch.empty(0)
        radius = self.pointcloud.get_radius if have_unstable else torch.empty(0)
        normal = self.pointcloud.get_normal if have_unstable else torch.empty(0)
        confidence = self.pointcloud.get_confidence if have_unstable else torch.empty(0)
        unstable_params = {
            "xyz": devF(xyz),
            "opacity": devF(opacity),
            "scales": devF(scales),
            "rotations": devF(rotations),
            "shs": devF(shs),
            "radius": devF(radius),
            "normal": devF(normal),
            "confidence": devF(confidence),
        }
        return unstable_params
    
    @property
    def dyna_params(self):
        have_unstable = self.get_dyna_num > 0
        xyz = self.dyna_pointcloud.get_xyz if have_unstable else torch.empty(0)
        opacity = self.dyna_pointcloud.get_opacity if have_unstable else torch.empty(0)
        scales = self.dyna_pointcloud.get_scaling if have_unstable else torch.empty(0)
        rotations = self.dyna_pointcloud.get_rotation if have_unstable else torch.empty(0)
        shs = self.dyna_pointcloud.get_features if have_unstable else torch.empty(0)
        radius = self.dyna_pointcloud.get_radius if have_unstable else torch.empty(0)
        normal = self.dyna_pointcloud.get_normal if have_unstable else torch.empty(0)
        confidence = self.dyna_pointcloud.get_confidence if have_unstable else torch.empty(0)
        dyna_params = {
            "xyz": devF(xyz),
            "opacity": devF(opacity),
            "scales": devF(scales),
            "rotations": devF(rotations),
            "shs": devF(shs),
            "radius": devF(radius),
            "normal": devF(normal),
            "confidence": devF(confidence),
        }
        return dyna_params

    @property
    def global_params_detach(self):
        unstable_params = self.unstable_params
        stable_params = self.stable_params
        for k in unstable_params:
            unstable_params[k] = unstable_params[k].detach()
        for k in stable_params:
            stable_params[k] = stable_params[k].detach()

        xyz = torch.cat([unstable_params["xyz"], stable_params["xyz"]])
        opacity = torch.cat([unstable_params["opacity"], stable_params["opacity"]])
        scales = torch.cat([unstable_params["scales"], stable_params["scales"]])
        rotations = torch.cat(
            [unstable_params["rotations"], stable_params["rotations"]]
        )
        shs = torch.cat([unstable_params["shs"], stable_params["shs"]])
        radius = torch.cat([unstable_params["radius"], stable_params["radius"]])
        normal = torch.cat([unstable_params["normal"], stable_params["normal"]])
        confidence = torch.cat(
            [unstable_params["confidence"], stable_params["confidence"]]
        )
        global_prams = {
            "xyz": xyz,
            "opacity": opacity,
            "scales": scales,
            "rotations": rotations,
            "shs": shs,
            "radius": radius,
            "normal": normal,
            "confidence": confidence,
        }
        return global_prams

    @property
    def static_params(self):
        unstable_params = self.unstable_params
        stable_params = self.stable_params

        xyz = torch.cat([unstable_params["xyz"], stable_params["xyz"]])
        opacity = torch.cat([unstable_params["opacity"], stable_params["opacity"]])
        scales = torch.cat([unstable_params["scales"], stable_params["scales"]])
        rotations = torch.cat(
            [unstable_params["rotations"], stable_params["rotations"]]
        )
        shs = torch.cat([unstable_params["shs"], stable_params["shs"]])
        radius = torch.cat([unstable_params["radius"], stable_params["radius"]])
        normal = torch.cat([unstable_params["normal"], stable_params["normal"]])
        confidence = torch.cat(
            [unstable_params["confidence"], stable_params["confidence"]]
        )
        global_prams = {
            "xyz": xyz,
            "opacity": opacity,
            "scales": scales,
            "rotations": rotations,
            "shs": shs,
            "radius": radius,
            "normal": normal,
            "confidence": confidence,
        }
        return global_prams
    
    @property
    def global_params(self):
        unstable_params = self.unstable_params
        stable_params = self.stable_params
        dyna_params = self.dyna_params

        xyz = torch.cat([unstable_params["xyz"], stable_params["xyz"], dyna_params["xyz"]])
        opacity = torch.cat([unstable_params["opacity"], stable_params["opacity"], dyna_params["opacity"]])
        scales = torch.cat([unstable_params["scales"], stable_params["scales"], dyna_params["scales"]])
        rotations = torch.cat(
            [unstable_params["rotations"], stable_params["rotations"], dyna_params["rotations"]]
        )
        shs = torch.cat([unstable_params["shs"], stable_params["shs"], dyna_params["shs"]])
        radius = torch.cat([unstable_params["radius"], stable_params["radius"], dyna_params["radius"]])
        normal = torch.cat([unstable_params["normal"], stable_params["normal"], dyna_params["normal"]])
        confidence = torch.cat(
            [unstable_params["confidence"], stable_params["confidence"], dyna_params["confidence"]]
        )
        global_prams = {
            "xyz": xyz,
            "opacity": opacity,
            "scales": scales,
            "rotations": rotations,
            "shs": shs,
            "radius": radius,
            "normal": normal,
            "confidence": confidence,
        }
        return global_prams


    @property
    def global_params_future(self):
        have_unstable = self.get_dyna_future_num > 0
        xyz = self.dyna_pointcloud_future.get_xyz if have_unstable else torch.empty(0)
        opacity = self.dyna_pointcloud_future.get_opacity if have_unstable else torch.empty(0)
        scales = self.dyna_pointcloud_future.get_scaling if have_unstable else torch.empty(0)
        rotations = self.dyna_pointcloud_future.get_rotation if have_unstable else torch.empty(0)
        shs = self.dyna_pointcloud_future.get_features if have_unstable else torch.empty(0)
        radius = self.dyna_pointcloud_future.get_radius if have_unstable else torch.empty(0)
        normal = self.dyna_pointcloud_future.get_normal if have_unstable else torch.empty(0)
        confidence = self.dyna_pointcloud_future.get_confidence if have_unstable else torch.empty(0)
        dyna_future_params = {
            "xyz": devF(xyz),
            "opacity": devF(opacity),
            "scales": devF(scales),
            "rotations": devF(rotations),
            "shs": devF(shs),
            "radius": devF(radius),
            "normal": devF(normal),
            "confidence": devF(confidence),
        }
        
        
        unstable_params = self.unstable_params
        stable_params = self.stable_params
        
        xyz = torch.cat([unstable_params["xyz"], stable_params["xyz"], dyna_future_params["xyz"]])
        opacity = torch.cat([unstable_params["opacity"], stable_params["opacity"], dyna_future_params["opacity"]])
        scales = torch.cat([unstable_params["scales"], stable_params["scales"], dyna_future_params["scales"]])
        rotations = torch.cat(
            [unstable_params["rotations"], stable_params["rotations"], dyna_future_params["rotations"]]
        )
        shs = torch.cat([unstable_params["shs"], stable_params["shs"], dyna_future_params["shs"]])
        radius = torch.cat([unstable_params["radius"], stable_params["radius"], dyna_future_params["radius"]])
        normal = torch.cat([unstable_params["normal"], stable_params["normal"], dyna_future_params["normal"]])
        confidence = torch.cat(
            [unstable_params["confidence"], stable_params["confidence"], dyna_future_params["confidence"]]
        )
        global_prams = {
            "xyz": xyz,
            "opacity": opacity,
            "scales": scales,
            "rotations": rotations,
            "shs": shs,
            "radius": radius,
            "normal": normal,
            "confidence": confidence,
        }
        return global_prams
    
    @property
    def get_pixel_num(self):
        return (
            self.frame_map["depth_map"].shape[0] * self.frame_map["depth_map"].shape[1]
        )

    @property
    def get_total_iter(self):
        return self.iter + self.time * self.gaussian_update_iter

    @property
    def get_stable_num(self):
        return self.stable_pointcloud.get_points_num

    @property
    def get_unstable_num(self):
        return self.pointcloud.get_points_num
    
    @property
    def get_dyna_num(self):
        return self.dyna_pointcloud.get_points_num

    @property
    def get_dyna_future_num(self):
        return self.dyna_pointcloud.get_points_num
    
    @property
    def get_total_num(self):
        return self.get_stable_num + self.get_unstable_num + self.get_dyna_num

    @property
    def get_curr_frame(self):
        return self.optimize_frames_ids[-1]

    @property
    def get_keyframe_num(self):
        return len(self.keyframe_list)


class MappingProcess(Mapping):
    def __init__(self, map_params, optimization_params, slam):
        super().__init__(map_params)
        self.recorder = Recorder(map_params.device_list[0])
        print("finish init")

        self.slam = slam
        # tracker 2 mapper
        self._tracker2mapper_call = slam._tracker2mapper_call
        self._tracker2mapper_frame_queue = slam._tracker2mapper_frame_queue

        # mapper 2 system
        self._mapper2system_call = slam._mapper2system_call
        self._mapper2system_map_queue = slam._mapper2system_map_queue
        self._mapper2system_tb_queue = slam._mapper2system_tb_queue
        self._mapper2system_requires = slam._mapper2system_requires

        # mapper 2 tracker
        self._mapper2tracker_call = slam._mapper2tracker_call
        self._mapper2tracker_map_queue = slam._mapper2tracker_map_queue

        self._requests = [False, False]  # [frame process, global optimization]
        self._stop = False
        self.input = {}
        self.output = {}
        self.processed_tick = []
        self.time = 0
        self.optimization_params = optimization_params
        self._end = slam._end
        self.max_frame_id = -1

        self.finish = mp.Condition()

    def set_input(self):
        self.frame_map["depth_map"] = self.input["depth_map"]
        self.frame_map["color_map"] = self.input["color_map"]
        self.frame_map["normal_map_c"] = self.input["normal_map_c"]
        self.frame_map["normal_map_w"] = self.input["normal_map_w"]
        self.frame_map["vertex_map_c"] = self.input["vertex_map_c"]
        self.frame_map["vertex_map_w"] = self.input["vertex_map_w"]
        self.frame = self.input["frame"]
        self.time = self.input["time"]
        self.last_send_time = -1

    def send_output(self):
        self.output = {
            "pointcloud": self.pointcloud,
            "stable_pointcloud": self.stable_pointcloud,
            "time": self.time,
            "iter": self.iter,
        }
        print("send output: ", self.time)
        self._mapper2system_map_queue.put(copy.deepcopy(self.output))
        self._mapper2system_requires[1] = True
        with self._mapper2system_call:
            self._mapper2system_call.notify()

    def release_receive(self):
        while (
            not self._tracker2mapper_frame_queue.empty()
            and self._tracker2mapper_frame_queue.qsize() > 1
        ):
            x = self._tracker2mapper_frame_queue.get()
            self.max_frame_id = max(self.max_frame_id, x["time"])
            print("release: ", x["time"])
            if x["time"] == -1:
                self.input = x
            else:
                del x

    def pack_map_to_tracker(self):
        map_info = {
            "frame": copy.deepcopy(self.frame),
            "global_params": self.global_params_detach,
            "frame_id": self.processed_tick[-1],
        }
        print("mapper send map {} to tracker".format(self.processed_tick[-1]))
        with self._mapper2tracker_call:
            self._mapper2tracker_map_queue.put(map_info)
            self._mapper2tracker_call.notify()

    def run(self):
        while True:
            print("map run...")
            with self._tracker2mapper_call:
                if self._tracker2mapper_frame_queue.qsize() == 0:
                    print("waiting tracker to wakeup")
                    self._tracker2mapper_call.wait()
                self.input = self._tracker2mapper_frame_queue.get()
                self.max_frame_id = max(self.max_frame_id, self.input["time"])

            # TODO: debug input is None
            if "time" in self.input and self.input["time"] == -1:
                del self.input
                break
            
            # run frame map update
            self.set_input()
            self.processed_tick.append(self.time)
            self.mapping(self.frame, self.frame_map, self.input["time"], self.optimization_params)
            self.pack_map_to_tracker()


        # self.release_receive()
        self.global_optimization(self.optimization_params)
        self.time = -1
        self.send_output()
        print("processed frames: ", self.optimize_frames_ids)
        print("keyframes: ", self.keyframe_ids)
        self._end[1] = 1
        with self._mapper2system_call:
            self._mapper2system_call.notify()

        with self.finish:
            print("mapper wating finish")
            self.finish.wait()
        print("map finish")

    def stop(self):
        with self.finish:
            self.finish.notify()
