import numpy as np
import math
import torch

from dynagslam.scene.cameras import Camera
from dynagslam.SLAM.utils import devF, devI

from diff_gaussian_rasterization_depth import (
    GaussianRasterizationSettings as GaussianRasterizationSettings_depth,
)
from diff_gaussian_rasterization_depth import (
    GaussianRasterizer as GaussianRasterizer_depth,
)

from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer



from dynagslam.utils.general_utils import (
    build_covariance_from_scaling_rotation,
    inverse_sigmoid
)


class Renderer:
    def setup_functions(self):
        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log

        self.covariance_activation = build_covariance_from_scaling_rotation

        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid

        self.rotation_activation = torch.nn.functional.normalize

    def __init__(self, args):
        self.L = args.L # Foureir series level
        self.dyna_start_iter = args.dyna_start_iter
        self.raster_settings = None
        self.rasterizer = None
        self.rasterizer_flow = None
        self.bg_color = devF(torch.tensor([0, 0, 0]))
        self.renderer_opaque_threshold = args.renderer_opaque_threshold
        self.renderer_normal_threshold = np.cos(
            np.deg2rad(args.renderer_normal_threshold)
        )
        self.scaling_modifier = 1.0
        self.renderer_depth_threshold = args.renderer_depth_threshold
        self.max_sh_degree = args.max_sh_degree
        self.color_sigma = args.color_sigma
        if args.active_sh_degree < 0:
            self.active_sh_degree = self.max_sh_degree
        else:
            self.active_sh_degree = args.active_sh_degree
        self.setup_functions()
        

    def get_scaling(self, scaling):
        return self.scaling_activation(scaling)

    def get_rotation(self, rotaion):
        return self.rotation_activation(rotaion)

    def get_covariance(self, scaling, rotaion, scaling_modifier=1):
        return self.covariance_activation(scaling, scaling_modifier, rotaion)

    def render(
        self,
        viewpoint_camera: Camera,
        gaussian_data,
        movinggs_idx,
        time,
        iter,
        tile_mask=None,
    ):
        if time == 0:
            render_flow=False
        else:
            render_flow=True
        
        tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
        tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)
        self.raster_settings = GaussianRasterizationSettings_depth(
            image_height=int(viewpoint_camera.image_height),
            image_width=int(viewpoint_camera.image_width),
            tanfovx=tanfovx,
            tanfovy=tanfovy,
            bg=self.bg_color,
            scale_modifier=self.scaling_modifier,
            viewmatrix=viewpoint_camera.world_view_transform,
            projmatrix=viewpoint_camera.full_proj_transform,
            sh_degree=self.active_sh_degree,
            campos=viewpoint_camera.camera_center,
            opaque_threshold=self.renderer_opaque_threshold,
            depth_threshold=self.renderer_depth_threshold,
            normal_threshold=self.renderer_normal_threshold,
            color_sigma=self.color_sigma,
            prefiltered=False,
            debug=False,
            cx=viewpoint_camera.cx,
            cy=viewpoint_camera.cy,
            T_threshold=0.0001,
        )
        self.rasterizer = GaussianRasterizer_depth(
            raster_settings=self.raster_settings
        )
        self.rasterizer_flow = GaussianRasterizer(
            raster_settings=self.raster_settings
        )
        
        means3D = gaussian_data["xyz"] #.clone() #[staticgs_idx]
        rotations = gaussian_data["rotations"] #.clone()
        
        opacity = gaussian_data["opacity"]
        scales = gaussian_data["scales"]     
        shs = gaussian_data["shs"]
        normal = gaussian_data["normal"]
        cov3D_precomp = None
        colors_precomp = None
        if tile_mask is None:
            tile_mask = devI(
                torch.ones(
                    (viewpoint_camera.image_height + 15) // 16,
                    (viewpoint_camera.image_width + 15) // 16,
                    dtype=torch.int32,
                )
            )
        
        render_results = self.rasterizer(
            means3D=means3D[:,0,:],
            opacities=opacity,
            shs=shs,
            colors_precomp=colors_precomp,
            scales=scales,
            rotations=rotations[:,0,:],
            cov3D_precomp=cov3D_precomp,
            normal_w=normal,
            tile_mask=tile_mask,
        )
        

        rendered_image = render_results[0]
        rendered_depth = render_results[1]
        color_index_map = render_results[2]
        depth_index_map = render_results[3]
        color_hit_weight = render_results[4]
        depth_hit_weight = render_results[5]
        T_map = render_results[6]

        render_normal = devF(torch.zeros_like(rendered_image))
        render_normal[:, depth_index_map[0] > -1] = normal[
            depth_index_map[depth_index_map > -1].long()
        ].permute(1, 0)
        
        results = {
            "render": rendered_image,
            "depth": rendered_depth,
            "normal": render_normal,
            "color_index_map": color_index_map,
            "depth_index_map": depth_index_map,
            "color_hit_weight": color_hit_weight,
            "depth_hit_weight": depth_hit_weight,
            "T_map": T_map,
            "flow": None
        }
        
        '''
        # render optical flow, the flow we estimate is from t to t-1
        if render_flow:
            # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
            screenspace_points = torch.zeros_like(means3D[:, 0, :], dtype=means3D.dtype, requires_grad=True, device="cuda") + 0
            try:
                screenspace_points.retain_grad()
            except:
                pass
            means2D = screenspace_points[:]
            
            #time_delta = 0.05/10
            shs = None
            colors_precomp = None
            flow = (means3D[:,1:2*L+1,:]*basis.unsqueeze(-1)).sum(1) # f(t) - f(t-1), this is 3D flow from t-1 to t estimated from t
            
            # compensate for the ego motion if the gt flow estimator doesn't know the ego camera is moving, 
            # comment out if the flow estimator knows the ego camera is moving
            flow = flow + viewpoint_camera.world_view_transform[3, :3] - self.old_camt

            focal_y = int(viewpoint_camera.image_height) / (2.0 * tanfovy)
            focal_x = int(viewpoint_camera.image_width) / (2.0 * tanfovx)
            tx, ty, tz = viewpoint_camera.world_view_transform[3, :3]
            viewmatrix = viewpoint_camera.world_view_transform.cuda()
            t = torch.matmul(means3D[:, 0, :], viewmatrix[:3, :3]) + viewmatrix[3, :3]    
            t = t.detach()
            # first term is normal projection, secend term is correction for flow along z dim
            flow[:, 0] = flow[:, 0] * focal_x / t[:, 2]  + flow[:, 2] * -(focal_x * t[:, 0]) / (t[:, 2]*t[:, 2])
            flow[:, 1] = flow[:, 1] * focal_y / t[:, 2]  + flow[:, 2] * -(focal_y * t[:, 1]) / (t[:, 2]*t[:, 2])
    
            colors_precomp = flow
            # only want to use the flow (colors_precomp) to update means3D so detach all other values including means3D directly from rasterization
            
            # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
            screenspace_points = torch.zeros_like(means3D[:, 0, :], dtype=means3D.dtype, requires_grad=True, device="cuda") + 0
            means2D = screenspace_points[:]
            try:
                screenspace_points.retain_grad()
            except:
                pass
            
            rendered_flow = self.rasterizer_flow(
                means3D = means3D[:,0,:].detach(),
                means2D = means2D.detach(),
                shs = shs,
                colors_precomp = colors_precomp,
                opacities = opacity.detach(),
                scales = scales.detach(),
                rotations = rotations[:,0,:].detach(),
                cov3D_precomp = cov3D_precomp)
            results['flow'] = -rendered_flow[0][:2, ...] #the reason for the minus sign, we need the flow from t to t-1 estimated from t
            '''
            
            
        self.old_camt = viewpoint_camera.world_view_transform[3, :3].detach()
        return results
