import os
from argparse import ArgumentParser

from utils.config_utils import read_config
parser = ArgumentParser(description="Training script parameters")
parser.add_argument("--config", type=str, default="configs/replica/office0.yaml")
args = parser.parse_args()
config_path = args.config
args = read_config(config_path)
os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(device) for device in args.device_list)
import torch
import json
from utils.camera_utils import loadCam
from arguments import DatasetParams, MapParams, OptimizationParams
from scene import Dataset
from SLAM.multiprocess.mapper_dyna_eval_sam_keti import Mapping
from SLAM.multiprocess.tracker import Tracker
from SLAM.utils import *
from SLAM.eval import eval_frame
from utils.general_utils import safe_state
from utils.monitor import Recorder
import matplotlib.pyplot as plt

torch.set_printoptions(4, sci_mode=False)


def main():
    # set visible devices
    time_recorder = Recorder(args.device_list[0])
    optimization_params = OptimizationParams(parser)
    dataset_params = DatasetParams(parser, sentinel=True)
    map_params = MapParams(parser)

    safe_state(args.quiet)
    optimization_params = optimization_params.extract(args)
    dataset_params = dataset_params.extract(args)
    map_params = map_params.extract(args)

    # Initialize dataset
    dataset = Dataset(
        dataset_params,
        shuffle=False,
        resolution_scales=dataset_params.resolution_scales,
    )

    record_mem = args.record_mem

    gaussian_map = Mapping(args, time_recorder)
    gaussian_map.create_workspace()
    gaussian_tracker = Tracker(args)
    # save config file
    prepare_cfg(args)
    # set time log
    tracker_time_sum = 0
    mapper_time_sum = 0
    
    # start SLAM
    psnr_total = 0
    psnr_dyna_total = 0
    ssim_total = 0
    lpips_total = 0
    semantic_old = None
    rgb_old = None
    timestamp_old = None
    dynaframe_count = 0
    for frame_id, frame_info in enumerate(dataset.scene_info.train_cameras):
        if dataset.scene_info.semantics is not None:
            semantic = dataset.scene_info.semantics[frame_id]
            semantic_eval = dataset.scene_info.semantics_eval[frame_id]
        else:
            semantic = None
        timestamp_curr = dataset.scene_info.timestamps[frame_id]
        if dataset.scene_info.flows is not None:
            flow = dataset.scene_info.flows[frame_id]
        else:
            flow = None
        curr_frame = loadCam(
            dataset_params, frame_id, frame_info, dataset_params.resolution_scales[0]
        )
        '''
        # eval for future extrapolation 
        frame_info_eval = dataset.scene_info.train_cameras[frame_id+10]
        curr_frame_eval = loadCam(
            dataset_params, frame_id+10, frame_info_eval, dataset_params.resolution_scales[0]
        )'''
        
        
        #eval for past interpolation
        frame_info_eval = dataset.scene_info.test_cameras[frame_id] #train_cameras[frame_id+1]
        curr_frame_eval = loadCam(
            dataset_params, frame_id, frame_info_eval, dataset_params.resolution_scales[0]
        )
        
        print("\n========== curr frame is: %d ==========\n" % frame_id)
        move_to_gpu(curr_frame)
        start_time = time.time()
        # tracker process
        frame_map = gaussian_tracker.map_preprocess(curr_frame, frame_id)      
        gaussian_tracker.tracking(curr_frame, frame_map, semantic, semantic_old)
        semantic_old = semantic
        tracker_time = time.time()
        tracker_consume_time = tracker_time - start_time
        time_recorder.update_mean("tracking", tracker_consume_time, 1)

        tracker_time_sum += tracker_consume_time
        print(f"[LOG] tracker cost time: {tracker_time - start_time}")

        mapper_start_time = time.time()

        new_poses = gaussian_tracker.get_new_poses()
        gaussian_map.update_poses(new_poses)
        
        # mapper process
        gaussian_map.mapping(curr_frame, curr_frame_eval, frame_map, frame_id, optimization_params, semantic, semantic_eval, flow, timestamp_curr, timestamp_old)

        timestamp_old = timestamp_curr
        gaussian_map.get_render_output(curr_frame)
        gaussian_tracker.update_last_status(
            curr_frame,
            gaussian_map.model_map["render_depth"],
            gaussian_map.frame_map["depth_map"],
            gaussian_map.model_map["render_normal"],
            gaussian_map.frame_map["normal_map_w"],
        )
        mapper_time = time.time()
        mapper_consume_time = mapper_time - mapper_start_time
        time_recorder.update_mean("mapping", mapper_consume_time, 1)

        mapper_time_sum += mapper_consume_time
        print(f"[LOG] mapper cost time: {mapper_time - tracker_time}")
        if record_mem:
            time_recorder.watch_gpu()
        # report eval loss
        if ((gaussian_map.time + 1) % gaussian_map.save_step == 0) or (
            gaussian_map.time == 0
        ):
            losses = eval_frame(
                gaussian_map,
                curr_frame,
                os.path.join(gaussian_map.save_path, "eval_render"),
                semantic,
                min_depth=gaussian_map.min_depth,
                max_depth=gaussian_map.max_depth,
                save_picture=True,
                run_pcd=False
            )
            #gaussian_map.save_model(save_data=True)
            if torch.sum(torch.tensor(semantic, dtype=torch.int64)) != 0:
                dynaframe_count += 1
            psnr_total += losses['psnr']
            mean_psnr = psnr_total/(frame_id+1)
            print("mean_psnr", mean_psnr)
            psnr_dyna_total += losses['psnr_dyna']
            if dynaframe_count != 0:
                mean_psnr_dyna = psnr_dyna_total/dynaframe_count #(frame_id+1)
                print("mean_psnr_dyna", mean_psnr_dyna)
            ssim_total += losses['ssim']
            mean_ssim = ssim_total/(frame_id+1)
            print("mean_ssim", mean_ssim)
            lpips_total += losses['lpips']
            mean_lpips = lpips_total/(frame_id+1)
            print("mean_lpips", mean_lpips)

        gaussian_map.time += 1
        move_to_cpu(curr_frame)
        torch.cuda.empty_cache()
    
    print("\n========== main loop finish ==========\n")
    print(
        "[LOG] stable num: {:d}, unstable num: {:d}".format(
            gaussian_map.get_stable_num, gaussian_map.get_unstable_num
        )
    )
    print("[LOG] processed frame: ", gaussian_map.optimize_frames_ids)
    print("[LOG] keyframes: ", gaussian_map.keyframe_ids)
    print("[LOG] mean tracker process time: ", tracker_time_sum / (frame_id + 1))
    print("[LOG] mean mapper process time: ", mapper_time_sum / (frame_id + 1))


if __name__ == "__main__":
    main()
