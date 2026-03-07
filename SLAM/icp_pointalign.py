import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from SLAM.utils import *
import matplotlib.pyplot as plt
import open3d as o3d



from GCNet.models import architectures, NgeNet, vote
from GCNet.models.utils import decode_config, npy2pcd, pcd2npy, execute_global_registration, \
                  npy2feat, setup_seed, get_blue, get_yellow, voxel_ds, normal, \
                  read_cloud, vis_plys
from easydict import EasyDict as edict
import copy
from GCNet.data import collate_fn



class NgeNet_pipeline():
    def __init__(self, ckpt_path, voxel_size, vote_flag, cuda=True):
        self.voxel_size_3dmatch = 0.025
        self.voxel_size = voxel_size
        self.scale = self.voxel_size / self.voxel_size_3dmatch
        self.cuda = cuda
        self.vote_flag = vote_flag
        config = self.prepare_config()
        self.neighborhood_limits = [38, 36, 35, 38]
        model = NgeNet(config)
        if self.cuda:
            model = model.cuda()
            model.load_state_dict(torch.load(ckpt_path))
        else:
            model.load_state_dict(
                torch.load(ckpt_path, map_location=torch.device('cpu')))
        self.model = model
        self.config = config
        self.model.eval()
    
    def prepare_config(self):
        config = decode_config(os.path.join('//home/blark/Desktop/2024/3dr/GCNet/configs', 'threedmatch.yaml'))
        config = edict(config)
        # config.first_subsampling_dl = self.voxel_size
        config.architecture = architectures[config.dataset]
        return config

    def prepare_inputs(self, source, target):
        src_pcd_input = pcd2npy(voxel_ds(copy.deepcopy(source), self.voxel_size))
        tgt_pcd_input = pcd2npy(voxel_ds(copy.deepcopy(target), self.voxel_size))

        src_pcd_input /= self.scale
        tgt_pcd_input /= self.scale

        src_feats = np.ones_like(src_pcd_input[:, :1])
        tgt_feats = np.ones_like(tgt_pcd_input[:, :1])

        src_pcd = normal(npy2pcd(src_pcd_input), radius=4*self.voxel_size_3dmatch, max_nn=30, loc=(0, 0, 0))
        tgt_pcd = normal(npy2pcd(tgt_pcd_input), radius=4*self.voxel_size_3dmatch, max_nn=30, loc=(0, 0, 0))
        src_normals = np.array(src_pcd.normals).astype(np.float32) 
        tgt_normals = np.array(tgt_pcd.normals).astype(np.float32)

        T = np.eye(4)
        coors = np.array([[0, 0], [1, 1]])
        src_pcd = pcd2npy(source)
        tgt_pcd = pcd2npy(target)

        pair = dict(
            src_points=src_pcd_input,
            tgt_points=tgt_pcd_input,
            src_feats=src_feats,
            tgt_feats=tgt_feats,
            src_normals=src_normals,
            tgt_normals=tgt_normals,
            transf=T,
            coors=coors,
            src_points_raw=src_pcd,
            tgt_points_raw=tgt_pcd)
        
        dict_inputs = collate_fn([pair], self.config, self.neighborhood_limits)
        if self.cuda:
            for k, v in dict_inputs.items():
                    if isinstance(v, list):
                        for i in range(len(v)):
                            dict_inputs[k][i] = dict_inputs[k][i].cuda()
                    else:
                        dict_inputs[k] = dict_inputs[k].cuda()
        
        return dict_inputs

    def pipeline(self, source, target, npts=20000):
        inputs = self.prepare_inputs(source, target)

        batched_feats_h, batched_feats_m, batched_feats_l = self.model(inputs)
        stack_points = inputs['points']
        stack_lengths = inputs['stacked_lengths']
        coords_src = stack_points[0][:stack_lengths[0][0]]
        coords_tgt = stack_points[0][stack_lengths[0][0]:]
        feats_src_h = batched_feats_h[:stack_lengths[0][0]]
        feats_tgt_h = batched_feats_h[stack_lengths[0][0]:]
        feats_src_m = batched_feats_m[:stack_lengths[0][0]]
        feats_tgt_m = batched_feats_m[stack_lengths[0][0]:]
        feats_src_l = batched_feats_l[:stack_lengths[0][0]]
        feats_tgt_l = batched_feats_l[stack_lengths[0][0]:]

        source_npy = coords_src.detach().cpu().numpy() * self.scale
        target_npy = coords_tgt.detach().cpu().numpy() * self.scale

        source_feats_h = feats_src_h[:, :-2].detach().cpu().numpy()
        target_feats_h = feats_tgt_h[:, :-2].detach().cpu().numpy()
        source_feats_m = feats_src_m.detach().cpu().numpy()
        target_feats_m = feats_tgt_m.detach().cpu().numpy()
        source_feats_l = feats_src_l.detach().cpu().numpy()
        target_feats_l = feats_tgt_l.detach().cpu().numpy() 

        source_overlap_scores = feats_src_h[:, -2].detach().cpu().numpy()
        target_overlap_scores = feats_tgt_h[:, -2].detach().cpu().numpy()
        source_scores = source_overlap_scores
        target_scores = target_overlap_scores

        npoints = npts
        if npoints > 0:
            if source_npy.shape[0] > npoints:
                p = source_scores / np.sum(source_scores)
                idx = np.random.choice(len(source_npy), size=npoints, replace=False, p=p)
                source_npy = source_npy[idx]
                source_feats_h = source_feats_h[idx]
                source_feats_m = source_feats_m[idx]
                source_feats_l = source_feats_l[idx]
            
            if target_npy.shape[0] > npoints:
                p = target_scores / np.sum(target_scores)
                idx = np.random.choice(len(target_npy), size=npoints, replace=False, p=p)
                target_npy = target_npy[idx]
                target_feats_h = target_feats_h[idx]
                target_feats_m = target_feats_m[idx]
                target_feats_l = target_feats_l[idx]
        
        if self.vote_flag:
            after_vote = vote(source_npy=source_npy, 
                            target_npy=target_npy, 
                            source_feats=[source_feats_h, source_feats_m, source_feats_l], 
                            target_feats=[target_feats_h, target_feats_m, target_feats_l], 
                            voxel_size=self.voxel_size * 2,
                            use_cuda=self.cuda)
            source_npy, target_npy, source_feats_npy, target_feats_npy = after_vote
        else:
            source_feats_npy, target_feats_npy = source_feats_h, target_feats_h
        source, target = npy2pcd(source_npy), npy2pcd(target_npy)
        source_feats, target_feats = npy2feat(source_feats_npy), npy2feat(target_feats_npy)
        pred_T, estimate = execute_global_registration(source=source,
                                                       target=target,
                                                       source_feats=source_feats,
                                                       target_feats=target_feats,
                                                       voxel_size=self.voxel_size*2)
        
        torch.cuda.empty_cache()
        return pred_T




def point2plane_loss(p_t0, p_t1, n_t0, reduce="mean"):
    loss = ((p_t1 - p_t0) * n_t0).sum(dim=-1)
    #loss = ((p_t1 - p_t0)).sum(dim=-1)
    if reduce == "mean":
        loss = (loss * loss).mean()
    else:
        loss = (loss * loss).sum()
    return loss


class ICP(nn.Module):
    def __init__(
        self,
        max_iter=3,
        damping=1e-6,
        distance_threshold=0.2,
        normal_threshold=20,
        verbose=False,
    ):
        super(ICP, self).__init__()

        self.max_iterations = max_iter
        self.distance_threshold = distance_threshold
        self.normal_threshold = np.cos(np.deg2rad(normal_threshold))
        self.damping = damping
        self.verbose = verbose
        
        # GCNet pointcloud alignment
        self.align_model = NgeNet_pipeline(
                    ckpt_path='//home/blark/Desktop/2024/3dr/GCNet/pretrained_model/3dmatch.pth', 
                    voxel_size=0.025, 
                    vote_flag=True,
                    cuda=True)

    def icp(self, pose10,vertex_t0,vertex_t1,normal_t0,normal_t1, K, semantic_old, semantic, depth_highgrad_t0, depth_highgrad_t1):
        mask0 = (vertex_t0[..., -1] > 0.0)
        mask0 = mask0 & (~semantic_old) & (~depth_highgrad_t0)
        
            
        # registration
        source_point, target_point, valid_mask = self.compute_residuals_jacobian(
                vertex_t0, vertex_t1, normal_t0, normal_t1, mask0, pose10, K,
                self.distance_threshold, self.normal_threshold, semantic, depth_highgrad_t1
            )
        source_point = torch.from_numpy(source_point)
        target_point = torch.from_numpy(target_point)
        source = o3d.geometry.PointCloud()
        source.points = o3d.utility.Vector3dVector(source_point)
        target = o3d.geometry.PointCloud()
        target.points = o3d.utility.Vector3dVector(target_point)
        pose_t1_t0 = self.align_model.pipeline(source, target, npts=source_point.shape[0])
        H,W = vertex_t0.shape[:2]
        valid_ratio = valid_mask.sum() / H / W
        return pose_t1_t0, valid_ratio, valid_mask
    

    @staticmethod
    def compute_residuals_jacobian(vertex0, vertex1, normal0, normal1, mask0, pose10, K, 
                                   distance_threshold, normal_threshold, semantic, depth_highgrad):
        """
        :param vertex0: vertex map 0
        :param vertex1: vertex map 1
        :param normal0: normal map 0
        :param normal1: normal map 1
        :param mask0: valid mask of template depth image
        :param pose10: current estimate of pose10
        :param K: intrinsics
        :return: residuals and Jacobians
        """
        surface_normals = normal1 / normal1.norm(dim=-1, keepdim=True).clamp(min=1e-8)  # Normalize each normal vector
        normalized_normals = (surface_normals + 1) / 2  # Shift and scale to [0, 1]
        
        R = pose10[:3, :3]
        t = pose10[:3, 3]
        H, W, C = vertex0.shape

        rot_vertex0_to1 = (R @ vertex0.view(-1, 3).permute(1, 0)).permute(1, 0).view(H, W, 3)
        vertex0_to1 = rot_vertex0_to1 + t[None, None, :]
        normal0_to1 = (R @ normal0.view(-1, 3).permute(1, 0)).permute(1, 0).view(H, W, 3)

        fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
        x_, y_, z_ = vertex0_to1[..., 0], vertex0_to1[..., 1], vertex0_to1[..., 2]  # [h, w]
        u_ = (x_ / z_) * fx + cx  # [h, w]
        v_ = (y_ / z_) * fy + cy  # [h, w]

        inviews = (u_ > 0) & (u_ < W-1) & (v_ > 0) & (v_ < H-1)
        # projective data association
        r_vertex1 = warp_features(vertex1, u_, v_)  # [h, w, 3]
        r_normal1 = warp_features(normal1, u_, v_)  # [h, w, 3]
        semantic_warp = warp_features(semantic.unsqueeze(-1).float(), u_, v_).squeeze(-1).bool()
        depth_highgrad_warp = warp_features(depth_highgrad.unsqueeze(-1).float(), u_, v_).squeeze(-1).bool()
        mask1 = r_vertex1[..., -1] > 0.
        
        mask1 = mask1 & (~semantic_warp) & (~depth_highgrad_warp)

        
        diff = vertex0_to1 - r_vertex1  # [h, w, 3]
        normal_diff_mask = torch.sum(normal0_to1 * r_normal1, dim=-1) > normal_threshold
        
        # point-to-plane residuals
        res = (r_normal1 * diff).sum(dim=-1)  # [h, w]
        # point-to-plane jacobians
        J_trs = r_normal1.view(-1, 3)  # [hw, 3]
        J_rot = -torch.bmm(J_trs.unsqueeze(dim=1), batch_skew(vertex0_to1.view(-1, 3))).squeeze()   # [hw, 3]

        # compose jacobians
        J_F_p = torch.cat((J_rot, J_trs), dim=-1).view(H, W, 6)  # follow the order of [rot, trs]  [hw, 1, 6]

        # occlusion
        occ = ~inviews | (diff.norm(p=2, dim=-1) > distance_threshold) 
        invalid_mask = ~mask0 | ~mask1
        return vertex0_to1[~invalid_mask].view(-1, 3).detach().cpu().numpy(), r_vertex1[~invalid_mask].view(-1,3).detach().cpu().numpy(), ~invalid_mask

    @staticmethod
    def compute_jtj(jac):
        # J in the dimension of (HW, C, 6)
        jacT = jac.transpose(-1, -2)  # [HW, 6, C]
        jtj = torch.bmm(jacT, jac).sum(0)  # [6, 6]
        return jtj  # [6, 6]

    @staticmethod
    def compute_jtr(jac, res):
        # J in the dimension of (HW, C, 6)
        # res in the dimension of [HW, C]
        jacT = jac.transpose(-1, -2)  # [HW, 6, C]
        jtr = torch.bmm(jacT, res.unsqueeze(-1)).sum(0)  # [6, 1]
        return jtr  # [6, 1]

    @staticmethod
    def GN_solver(JtJ, JtR, pose0, damping=1e-6):
        # Add a small diagonal damping. Without it, the training becomes quite unstable
        # Do not see a clear difference by removing the damping in inference though
        Hessian = lev_mar_H(JtJ, damping)
        # Hessian = JtJ
        updated_pose = forward_update_pose(Hessian, JtR, pose0)

        return updated_pose


def warp_features(Feat, u, v, mode="nearest"):
    """
    Warp the feature map (F) w.r.t. the grid (u, v). This is the non-batch version
    """
    assert len(Feat.shape) == 3
    H, W, C = Feat.shape
    u_norm = u / ((W - 1) / 2) - 1  # [h, w]
    v_norm = v / ((H - 1) / 2) - 1  # [h, w]
    uv_grid = torch.cat((u_norm.view(1, H, W, 1), v_norm.view(1, H, W, 1)), dim=-1)
    Feat_warped = F.grid_sample(
        Feat.unsqueeze(0).permute(0, 3, 1, 2),
        uv_grid,
        mode=mode,
        padding_mode="border",
        align_corners=True,
    ).squeeze()
    if Feat.shape[-1] == 1:
        Feat_warped = Feat_warped.unsqueeze(0)
    return Feat_warped.permute(1, 2, 0)


def compute_vertex(depth, K):
    H, W = depth.shape
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    device = depth.device

    i, j = torch.meshgrid(
        torch.linspace(0, W - 1, W), torch.linspace(0, H - 1, H)
    )  # pytorch's meshgrid has indexing='ij'
    i = i.t().to(device)  # [h, w]
    j = j.t().to(device)  # [h, w]

    vertex = (
        torch.stack([(i - cx) / fx, (j - cy) / fy, torch.ones_like(i)], -1).to(device)
        * depth[..., None]
    )  # [h, w, 3]
    return vertex


def compute_normal(vertex_map):
    """Calculate the normal map from a depth map
    :param the input depth image
    -----------
    :return the normal map
    """
    H, W, C = vertex_map.shape
    img_dx, img_dy = feature_gradient(vertex_map, normalize_gradient=False)  # [h, w, 3]

    normal = torch.cross(img_dx.view(-1, 3), img_dy.view(-1, 3))
    normal = normal.view(H, W, 3)  # [h, w, 3]

    mag = torch.norm(normal, p=2, dim=-1, keepdim=True)
    normal = normal / (mag + 1e-8)

    # filter out invalid pixels
    depth = vertex_map[:, :, -1]
    # 0.5 and 5.
    invalid_mask = (depth <= depth.min()) | (depth >= depth.max())
    zero_normal = torch.zeros_like(normal)
    normal = torch.where(invalid_mask[..., None], zero_normal, normal)

    return normal


def feature_gradient(img, normalize_gradient=True):
    """Calculate the gradient on the feature space using Sobel operator
    :param the input image
    -----------
    :return the gradient of the image in x, y direction
    """
    H, W, C = img.shape
    # to filter the image equally in each channel
    wx = (
        torch.FloatTensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
        .view(1, 1, 3, 3)
        .type_as(img)
    )
    wy = (
        torch.FloatTensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
        .view(1, 1, 3, 3)
        .type_as(img)
    )

    img_permuted = img.permute(2, 0, 1).view(-1, 1, H, W)  # [c, 1, h, w]
    img_pad = F.pad(img_permuted, (1, 1, 1, 1), mode="replicate")
    img_dx = (
        F.conv2d(img_pad, wx, stride=1, padding=0).squeeze().permute(1, 2, 0)
    )  # [h, w, c]
    img_dy = (
        F.conv2d(img_pad, wy, stride=1, padding=0).squeeze().permute(1, 2, 0)
    )  # [h, w, c]

    if normalize_gradient:
        mag = torch.sqrt((img_dx**2) + (img_dy**2) + 1e-8)
        img_dx = img_dx / mag
        img_dy = img_dy / mag

    return img_dx, img_dy  # [h, w, c]


def batch_skew(w):
    """Generate a batch of skew-symmetric matrices.

        function tested in 'test_geometry.py'

    :input
    :param skew symmetric matrix entry Bx3
    ---------
    :return
    :param the skew-symmetric matrix Bx3x3
    """
    B, D = w.shape
    assert D == 3
    o = torch.zeros(B).type_as(w)
    w0, w1, w2 = w[:, 0], w[:, 1], w[:, 2]
    return torch.stack((o, -w2, w1, w2, o, -w0, -w1, w0, o), 1).view(B, 3, 3)


def lev_mar_H(JtWJ, damping):
    # Add a small diagonal damping. Without it, the training becomes quite unstable
    # Do not see a clear difference by removing the damping in inference though
    diag_mask = torch.eye(6).to(JtWJ)
    diagJtJ = diag_mask * JtWJ
    traceJtJ = torch.sum(diagJtJ)
    damping = 0.001
    epsilon = (traceJtJ * damping) * diag_mask
    Hessian = JtWJ + epsilon
    return Hessian


def forward_update_pose(H, Rhs, pose):
    """
    :param H:
    :param Rhs:
    :param pose:
    :return:
    """
    xi = least_square_solve(H, Rhs).squeeze()
    #xi[5] = 0.0
    #xi[3] = 0.0
    pose = exp_se3(xi) @ pose
    return pose


def exp_se3(xi):
    """
    :param x: Cartesian vector of Lie Algebra se(3)
    :return: exponential map of x
    """
    w = xi[:3].squeeze()  # rotation
    v = xi[3:6].squeeze()  # translation
    w_hat = torch.tensor(
        [[0.0, -w[2], w[1]], [w[2], 0.0, -w[0]], [-w[1], w[0], 0.0]]
    ).to(xi)
    w_hat_second = torch.mm(w_hat, w_hat).to(xi)

    theta = torch.norm(w)
    theta_2 = theta**2
    theta_3 = theta**3
    sin_theta = torch.sin(theta)
    cos_theta = torch.cos(theta)
    eye_3 = torch.eye(3).to(xi)

    eps = 1e-8

    if theta <= eps:
        e_w = eye_3
        j = eye_3
    else:
        e_w = (
            eye_3
            + w_hat * sin_theta / theta
            + w_hat_second * (1.0 - cos_theta) / theta_2
        )
        k1 = (1 - cos_theta) / theta_2
        k2 = (theta - sin_theta) / theta_3
        j = eye_3 + k1 * w_hat + k2 * w_hat_second

    T = torch.eye(4).to(xi)
    T[:3, :3] = e_w
    T[:3, 3] = torch.mv(j, v)
    # T[:3, 3] = v

    return T


def invH(H):
    """Generate (H+damp)^{-1}, with predicted damping values
    :param approximate Hessian matrix JtWJ
    -----------
    :return the inverse of Hessian
    """
    # GPU is much slower for matrix inverse when the size is small (compare to CPU)
    # works (50x faster) than inversing the dense matrix in GPU
    if H.is_cuda:
        invH = torch.inverse(H.cpu()).cuda()
    else:
        invH = torch.inverse(H)
    return invH


def least_square_solve(H, Rhs):
    """
    Solve for JTJ @ xi = -JTR
    """
    inv_H = invH(H)  # [B, 6, 6] square matrix
    xi = -inv_H @ Rhs
    return xi


class ImagePyramids(nn.Module):
    """ Construct the pyramids in the image / depth space
    """
    def __init__(self, scales, pool='avg'):
        super(ImagePyramids, self).__init__()
        if pool == 'avg':
            self.multiscales = [nn.AvgPool2d(1<<i, 1<<i) for i in scales]
        elif pool == 'max':
            self.multiscales = [nn.MaxPool2d(1<<i, 1<<i) for i in scales]
        else:
            raise NotImplementedError()

    def forward(self, x):
        if x.dtype == torch.bool:
            x = x.to(torch.float32)
            x_out = [f(x).to(torch.bool) for f in self.multiscales]
        else:
            x_out = [f(x) for f in self.multiscales]
        return x_out

class IcpTracker:
    def __init__(self, args):
        self.icp_trackers = []
        self.icp_downscales = args.icp_downscales
        self.icp_warmup_frames = args.icp_warmup_frames
        self.icp_use_model_depth = args.icp_use_model_depth
        for iters in args.icp_downscale_iters:
            self.icp_trackers.append(
                ICP(
                    iters,
                    distance_threshold=args.icp_distance_threshold,
                    normal_threshold=args.icp_normal_threshold,
                    damping=args.icp_damping,
                    verbose=args.verbose,
                )
            )
        
        self.depth_pyramid_builder = ImagePyramids(list(range(len(self.icp_downscales)-1,-1, -1)), "max")
        self.icp_sample_distance_threshold = args.icp_sample_distance_threshold
        self.icp_sample_normal_threshold = args.icp_sample_normal_threshold
        self.icp_fail_threshold = args.icp_fail_threshold

        self.normal_pyramid_t0 = None
        self.vertex_pyramid_t0 = None
        self.verbose = args.verbose

        self.K = None
        
    def update_curr_status(self, depth_t1, K):
        if self.K is None:
            self.K = K
        self.depth_t1 = depth_t1
        self.vertex_pyramid_t1 = build_vertex_pyramid(depth_t1, self.depth_pyramid_builder, self.K)
        self.normal_pyramid_t1 = build_normal_pyramid(self.vertex_pyramid_t1)

    def move_last_status(self):
        self.vertex_pyramid_t0 = self.vertex_pyramid_t1
        self.normal_pyramid_t0 = self.normal_pyramid_t1
        self.last_model_depth = self.depth_t1

    def update_last_status(
        self, frame, render_depth, frame_depth, render_normal, frame_normal
    ):

        intrinsic = frame.get_intrinsic
        normal_mask = (
            1 - F.cosine_similarity(render_normal, frame_normal, dim=-1)
        ) > self.icp_sample_normal_threshold
        depth_filling_mask = (
            (
                torch.abs(render_depth - frame_depth)
                > self.icp_sample_distance_threshold
            )[..., 0]
            | (render_depth == 0)[..., 0]
            | (normal_mask)
        ) & (frame_depth > 0)[..., 0]

        render_depth[depth_filling_mask] = frame_depth[depth_filling_mask]
        self.last_model_depth = frame_depth #render_depth

    def predict_pose(self, frame, semantic, semantic_old):
        K = frame["K"]
        frame_id = frame["frame_id"]
        
        semantic_old = torch.tensor(semantic_old).to(self.vertex_pyramid_t0[0].device)
        semantic = torch.tensor(semantic).to(self.vertex_pyramid_t0[0].device)
        if self.vertex_pyramid_t0 is None:
            pose_t1_t0 = np.eye(4)
            self.K = K
        else:
            if self.icp_use_model_depth and frame_id >= self.icp_warmup_frames:
                # calculate depth gradient
                depth_y = self.last_model_depth.squeeze(-1).diff(dim=0)
                depth_x = self.last_model_depth.squeeze(-1).diff(dim=1)
                # Align dimensions by cropping the larger dimension
                min_height = min(depth_y.shape[0], depth_x.shape[0])  # Handle height alignment
                min_width = min(depth_y.shape[1], depth_x.shape[1])   # Handle width alignment
                # Crop both gradients to ensure alignment
                depth_y_cropped = depth_y[:min_height, :min_width]
                depth_x_cropped = depth_x[:min_height, :min_width]
                depth_gradient_t0 = torch.sqrt(depth_x_cropped**2 + depth_y_cropped**2)
                depth_highgrad_t0_temp = depth_gradient_t0 > 0.01
                depth_highgrad_t0 = torch.ones((depth_highgrad_t0_temp.shape[0]+1, depth_highgrad_t0_temp.shape[1]+1)).bool().to(depth_highgrad_t0_temp.device)
                depth_highgrad_t0[:-1, :-1] = depth_highgrad_t0_temp
                
                depth_y = self.depth_t1.squeeze(-1).diff(dim=0)
                depth_x = self.depth_t1.squeeze(-1).diff(dim=1)
                # Align dimensions by cropping the larger dimension
                min_height = min(depth_y.shape[0], depth_x.shape[0])  # Handle height alignment
                min_width = min(depth_y.shape[1], depth_x.shape[1])   # Handle width alignment
                # Crop both gradients to ensure alignment
                depth_y_cropped = depth_y[:min_height, :min_width]
                depth_x_cropped = depth_x[:min_height, :min_width]
                depth_gradient_t1 = torch.sqrt(depth_x_cropped**2 + depth_y_cropped**2)
                depth_highgrad_t1_temp = depth_gradient_t1 > 0.01 
                depth_highgrad_t1 = torch.ones((depth_highgrad_t1_temp.shape[0]+1, depth_highgrad_t1_temp.shape[1]+1)).bool().to(depth_highgrad_t1_temp.device)
                depth_highgrad_t1[:-1, :-1] = depth_highgrad_t1_temp
                
                
                self.vertex_pyramid_t0 = build_vertex_pyramid(self.last_model_depth, self.depth_pyramid_builder, self.K)                
                self.normal_pyramid_t0 = build_normal_pyramid(self.vertex_pyramid_t0)
            pose_t1_t0 = devF(torch.from_numpy(np.eye(4)))
            levels = len(self.icp_downscales)
            for level in range(levels):
                downscale = self.icp_downscales[level]
                semantic_down = F.interpolate(semantic.float().unsqueeze(0).unsqueeze(0), scale_factor=downscale, mode='bilinear', align_corners=False).squeeze(0).squeeze(0).bool()
                semantic_old_down = F.interpolate(semantic_old.float().unsqueeze(0).unsqueeze(0), scale_factor=downscale, mode='bilinear', align_corners=False).squeeze(0).squeeze(0).bool()
                
                depth_highgrad_t0_down = F.interpolate(depth_highgrad_t0.float().unsqueeze(0).unsqueeze(0), scale_factor=downscale, mode='bilinear', align_corners=False).squeeze(0).squeeze(0).bool()
                depth_highgrad_t1_down = F.interpolate(depth_highgrad_t1.float().unsqueeze(0).unsqueeze(0), scale_factor=downscale, mode='bilinear', align_corners=False).squeeze(0).squeeze(0).bool()
                
                
                K_downscale = K * downscale
                K_downscale[2,2] = 1.0
                vertex_t0 = self.vertex_pyramid_t0[level]
                vertex_t1 = self.vertex_pyramid_t1[level]
                normal_t0 = self.normal_pyramid_t0[level]
                normal_t1 = self.normal_pyramid_t1[level]
                pose_t1_t0, valid_ratio, valid_mask = self.icp_trackers[level].icp(
                    pose_t1_t0, vertex_t0, vertex_t1, normal_t0, normal_t1,
                    K_downscale, semantic_old_down, semantic_down, depth_highgrad_t0_down, depth_highgrad_t1_down 
                )
            
            
        pose_t1_t0_pytorch = torch.tensor(pose_t1_t0).cuda().float()
        p2ploss = point2plane_loss(self.vertex_pyramid_t0[-1], 
                                   self.vertex_pyramid_t1[-1] @ pose_t1_t0_pytorch[:3,:3].T + pose_t1_t0_pytorch[:3, 3], 
                                   self.normal_pyramid_t0[-1], 
                                   )
        tracking_success = True
        if p2ploss > self.icp_fail_threshold:
            tracking_success = False
        return pose_t1_t0, tracking_success
