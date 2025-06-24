import argparse
import time
import torch
import numpy as np
import math
from EGNet.utils.data import registration_collate_fn_stack_mode
from EGNet.utils.torch import to_cuda, release_cuda
from EGNet.utils.open3d import make_open3d_point_cloud, get_color, draw_geometries
from EGNet.utils.registration import compute_registration_error
import sys
from config import make_cfg
from model import create_model
import yaml
import open3d as o3d

def make_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src_file", required=True, help="src point cloud numpy file")
    parser.add_argument("--ref_file", required=True, help="src point cloud numpy file")
    parser.add_argument("--gt_file", required=True, help="ground-truth transformation file")
    parser.add_argument("--weights", required=True, help="model weights file")
    return parser


def load_data(args):
    src_points = np.load(args.src_file)
    ref_points = np.load(args.ref_file)
    src_feats = np.ones_like(src_points[:, :1])
    ref_feats = np.ones_like(ref_points[:, :1])

    data_dict = {
        "ref_points": ref_points.astype(np.float32),
        "src_points": src_points.astype(np.float32),
        "ref_feats": ref_feats.astype(np.float32),
        "src_feats": src_feats.astype(np.float32),
    }

    if args.gt_file is not None:
        transform = np.load(args.gt_file)
        data_dict["transform"] = transform.astype(np.float32)

    return data_dict


def main(num):
    # parser = make_parser()
    # args = parser.parse_args()

    cfg = make_cfg()

    # prepare data
    # data_dict = load_data(args)


    src_points = np.load('../../data/demo/Fudan/'+num+'Vscaled_spa0015_BeforeRigid.npy')  #09
    ref_points = np.load('../../data/demo/Fudan/'+num+'CTscaled_spa0015.npy')

    state_dict = torch.load('../../output/geotransformer.3dmatch.stage4.gse.k3.max.oacl.stage2.sinkhorn/snapshots/epoch-10-lightweight-TRE4.44-RRE4.27.pth.tar')

    src_feats = np.ones_like(src_points[:, :1])
    ref_feats = np.ones_like(ref_points[:, :1])

    data_dict = {
        "ref_points": ref_points.astype(np.float32),
        "src_points": src_points.astype(np.float32),
        "ref_feats": ref_feats.astype(np.float32),
        "src_feats": src_feats.astype(np.float32),
    }
    transform = np.load('../../data/demo/Fudan/transform_'+num+'.npy')
    data_dict["transform"] = transform.astype(np.float32)


    neighbor_limits = [42, 28, 27, 29]

    data_dict = registration_collate_fn_stack_mode(
        [data_dict], cfg.backbone.num_stages, cfg.backbone.init_voxel_size, cfg.backbone.init_radius, neighbor_limits
    )
    # prepare model

    model = create_model(cfg).cuda()
    model.load_state_dict(state_dict["model"])
    model.training = False
    # prediction
    data_dict = to_cuda(data_dict)
    since = time.time()
    output_dict = model(data_dict)
    time_elapsed = time.time() - since
    data_dict = release_cuda(data_dict)
    output_dict = release_cuda(output_dict)

    # get results
    ref_points = output_dict["ref_points"]
    src_points = output_dict["src_points"]
    estimated_transform = output_dict["estimated_transform"]
    transform = data_dict["transform"]
    # visualization
    ref_pcd = make_open3d_point_cloud(ref_points)
    ref_pcd.estimate_normals()
    ref_pcd.paint_uniform_color(get_color("custom_yellow"))
    src_pcd = make_open3d_point_cloud(src_points)
    src_pcd.estimate_normals()
    src_pcd.paint_uniform_color(get_color("custom_blue"))
    # draw_geometries(ref_pcd, src_pcd)
    src_pcd = src_pcd.transform(estimated_transform)
    # draw_geometries(ref_pcd, src_pcd)

    # output_file_path = '../../data/demo/Suzhou/'+num+'transformpre.txt'

    # 保存到文本文件
    # np.savetxt(output_file_path, estimated_transform, fmt='%f')

    # compute error
    rre, rte = compute_registration_error(transform, estimated_transform)
    # print(transform)
    # print(estimated_transform)

    output_folder = "Results/val-Fudan/"
    # output_folder = "ResultsAB/val-Fudan/"
    file_name = "pre_transforms" + num + ".txt"
    os.makedirs(output_folder, exist_ok=True)
    with open(os.path.join(output_folder, file_name), 'w') as file:
        for i, transform2 in enumerate(estimated_transform):
            file.write(f"{transform2}\n")
    src_pcd_after = src_pcd
    ply_file_name = f"transformed{num}.ply"
    ply_file_path = os.path.join(output_folder, ply_file_name)
    o3d.io.write_point_cloud(ply_file_path, src_pcd_after)

    return rre,rte * 10,time_elapsed

if __name__ == "__main__":


    # nums = [f"{i:02}" for i in range(9, 10)]
    nums = [f"{i:02}" for i in [1,2,3,4,5,6,7,8,9,11]]

    import statistics

    rre_total, rte_total, time_total, = 0, 0, 0
    RRElist, RTElist, Timelist = [], [], []
    rre_values = []
    rte_values = []
    time_values = []
    for num in nums:
        rre, rte, Time = main(num)
        rre_values.append(rre)
        rte_values.append(rte)
        time_values.append(Time)

        rre_total += rre
        rte_total += rte
        time_total += Time

        print(" num", num,  f" RRE(deg): {rre:.2f}, RTE(mm): {rte:.2f}, Time(s): {Time:.2f}")

        RRElist.append(rre)
        RTElist.append(rte)
        Timelist.append(Time)

    RREmean = np.mean(RRElist)
    RTEmean = np.mean(RTElist)
    Timemean = np.mean(Timelist)

    RREstd = np.std(RRElist)
    RTEstd = np.std(RTElist)
    Timestd = np.std(Timelist)

    print(f"Average RRE(deg): {RREmean:.2f}±{RREstd:.2f}")
    print(f"Average RTE(mm): {RTEmean:.2f}±{RTEstd:.2f}")
    print(f"Average Time(s): {Timemean:.2f}±{Timestd:.2f}")
