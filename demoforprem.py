import argparse
import time
import torch
import numpy as np
import math
from geotransformer.utils.data import registration_collate_fn_stack_mode
from geotransformer.utils.torch import to_cuda, release_cuda
from geotransformer.utils.open3d import make_open3d_point_cloud, get_color, draw_geometries
from geotransformer.utils.registration import compute_registration_error

from config import make_cfg
from model import create_model
import yaml
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '3'
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


def main(num,epoch):
    # parser = make_parser()
    # args = parser.parse_args()

    cfg = make_cfg()


    src_points = np.load('../../data/demo/DePoLL-prem/video_reconstructionBeforeRigid_scaled100_spa0015_'+num+'.npy')
    ref_points = np.load('../../data/demo/DePoLL-prem/surface_CT_scaled100_spa0015_'+num+'.npy')

    state_dict = torch.load('../../output/geotransformer.3dmatch.stage4.gse.k3.max.oacl.stage2.sinkhorn/snapshots/epoch-10-lightweight-TRE4.44-RRE4.27.pth.tar')
   

    src_feats = np.ones_like(src_points[:, :1])
    ref_feats = np.ones_like(ref_points[:, :1])

    data_dict = {
        "ref_points": ref_points.astype(np.float32),
        "src_points": src_points.astype(np.float32),
        "ref_feats": ref_feats.astype(np.float32),
        "src_feats": src_feats.astype(np.float32),
    }
    transform = np.load('../../data/demo/gtscaled100_' + num + '_V2S.npy')
    data_dict["transform"] = transform.astype(np.float32)

    neighbor_limits = [37, 32, 30, 34] 
   
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
    transformed_pcd = src_pcd.transform(estimated_transform)
    # o3d.io.write_point_cloud("../../RigidResults/val-prem/transformed"+num+".ply", transformed_pcd)
    # draw_geometries(ref_pcd, src_pcd)

    # compute error

    rre, rte = compute_registration_error(transform, estimated_transform)

    # output_folder = "Results/val-prem/"
    output_folder = "ResultsAB/val-prem/"
    file_name = "pre_transforms" + num + ".txt"
    os.makedirs(output_folder, exist_ok=True)
    with open(os.path.join(output_folder, file_name), 'w') as file:
        for i, transform2 in enumerate(estimated_transform):
            file.write(f"{transform2}\n")
    src_pcd_after = src_pcd
    ply_file_name = f"transformed{num}.ply"
    ply_file_path = os.path.join(output_folder, ply_file_name)
    o3d.io.write_point_cloud(ply_file_path, src_pcd_after)

    spacing_ref = [0.8125, 0.8125, 1]
    with open(r"../../Reg/markersref.yml", "r") as f:
        file = f.read()
    content = yaml.load(file, yaml.FullLoader)
    # print(content)
    coorball = np.zeros([45, 3])
    coorclip = np.zeros([15, 3])

    databall = content['databall']
    databall_coordinatesref = np.array(databall).reshape(content['rowsball'], content['colsball'])
    for i in range(0, 45):
        coorball[i][0] = databall_coordinatesref[0][i]
        coorball[i][1] = databall_coordinatesref[1][i]
        coorball[i][2] = databall_coordinatesref[2][i]
    dataclip = content['dataclip']
    dataclip_coordinatesref = np.array(dataclip).reshape(content['rowsclip'], content['colsclip'])
    for i in range(0, 15):
        coorclip[i][0] = dataclip_coordinatesref[0][i]
        coorclip[i][1] = dataclip_coordinatesref[1][i]
        coorclip[i][2] = dataclip_coordinatesref[2][i]
    dataref = np.concatenate((coorball, coorclip), axis=0)

    ##############ORG
    with open("../../Reg/markersnew" + num + ".yml", "r") as f:
        file = f.read()
    content = yaml.load(file, yaml.FullLoader)
    # print(content)
    coorball = np.zeros([45, 3])
    coorclip = np.zeros([15, 3])
    databall = content['databall']
    databall_coordinates = np.array(databall).reshape(content['rowsball'], content['colsball'])
    for i in range(0, 45):
        coorball[i][0] = databall_coordinates[0][i]
        coorball[i][1] = databall_coordinates[1][i]
        coorball[i][2] = databall_coordinates[2][i]
    # print("databall_coordinates is: \n",databall_coordinates)
    dataclip1 = content['dataclip']
    dataclip_coordinates = np.array(dataclip1).reshape(content['rowsclip'], content['colsclip'])
    for i in range(0, 15):
        coorclip[i][0] = dataclip_coordinates[0][i]
        coorclip[i][1] = dataclip_coordinates[1][i]
        coorclip[i][2] = dataclip_coordinates[2][i]
    # print("dataclip_coordinates is: \n",dataclip_coordinates)
    data = np.concatenate((coorball, coorclip), axis=0)
    #########################金标准变换矩阵
    # coordinate transforms
    sortdataappend = np.ones([60, 4])
    with open(r"../../Reg/Mnew" + num + ".yml", "r") as f:
        file = f.read()
    content = yaml.load(file, yaml.FullLoader)
    transform = content["data"]
    transform = np.array(transform).reshape(content['cols'], content['rows'])
    # print("gold",transform)
    for i in range(0, 60):
        sortdataappend[i] = np.append(data[i], 1)
        sortdataappend[i] = transform @ sortdataappend[i]
        data[i] = np.delete(sortdataappend[i], -1)
    # associations
    with open(r"../../Reg/associationsnew" + num + ".yml", "r") as f:
        file = f.read()
    content = yaml.load(file, yaml.FullLoader)
    sort = content["data"]
    sortdata = np.ones([60, 3])
    for i in range(0, 60):
        sortdata[i][0] = data[sort[i] - 1][0]
        sortdata[i][1] = data[sort[i] - 1][1]
        sortdata[i][2] = data[sort[i] - 1][2]
    #############################################
    # TRE
    diff = dataref - sortdata
    diff = diff * spacing_ref
    diff = torch.Tensor(diff)
    diff_clip = diff[-15:]
    TRE = diff_clip.pow(2).sum(1).sqrt()
    TRE = TRE.mean()

    ############################## pre
    ##################1#########################
    with open("../../Reg/markersnew" + num + ".yml", "r") as f:
        file = f.read()
    content = yaml.load(file, yaml.FullLoader)
    # print(content)
    coorball = np.zeros([45, 3])
    coorclip = np.zeros([15, 3])
    databall = content['databall']
    databall_coordinates = np.array(databall).reshape(content['rowsball'], content['colsball'])
    for i in range(0, 45):
        coorball[i][0] = databall_coordinates[0][i]
        coorball[i][1] = databall_coordinates[1][i]
        coorball[i][2] = databall_coordinates[2][i]
    # print("databall_coordinates is: \n",databall_coordinates)
    dataclip1 = content['dataclip']
    dataclip_coordinates = np.array(dataclip1).reshape(content['rowsclip'], content['colsclip'])
    for i in range(0, 15):
        coorclip[i][0] = dataclip_coordinates[0][i]
        coorclip[i][1] = dataclip_coordinates[1][i]
        coorclip[i][2] = dataclip_coordinates[2][i]
    # print("dataclip_coordinates is: \n",dataclip_coordinates)
    data = np.concatenate((coorball, coorclip), axis=0)
    #########################变换矩阵
    # coordinate transforms
    sortdataappend = np.ones([60, 4])
    # with open("../../Reg/Mpredict" + case + ".yml", "r") as f:
    #     file = f.read()
    # content = yaml.load(file, yaml.FullLoader)
    transform = np.array(estimated_transform)
    transform[:, 3] *= 100
    for i in range(0, 60):
        sortdataappend[i] = np.append(data[i], 1)
        sortdataappend[i] = transform @ sortdataappend[i]
        data[i] = np.delete(sortdataappend[i], -1)

    # associations
    with open(r"../../Reg/associationsnew" + num + ".yml", "r") as f:
        file = f.read()
    content = yaml.load(file, yaml.FullLoader)
    sort = content["data"]
    sortdata = np.ones([60, 3])
    for i in range(0, 60):
        sortdata[i][0] = data[sort[i] - 1][0]
        sortdata[i][1] = data[sort[i] - 1][1]
        sortdata[i][2] = data[sort[i] - 1][2]

    diff = dataref - sortdata
    diff = torch.Tensor(diff)
    diff_clip = diff[-15:]
    TREpre = diff_clip.pow(2).sum(1).sqrt()
    TREpre = TREpre.mean()

    diff_sum = abs(TRE - TREpre)


    return rre,rte * 10,time_elapsed,diff_sum

if __name__ == "__main__":


    nums = ['02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12', '13']
    # nums = ['10']
    TRElist, RRElist, RTElist, Timelist = [], [], [], []
    rre_values = []
    rte_values = []
    time_values = []
    tre_values = []
    epoch=3
    rre_total, rte_total, time_total, tre_total = 0, 0, 0, 0
    for num in nums:
        rre, rte, Time, diff_sum = main(num,epoch)
        rre_values.append(rre)
        rte_values.append(rte)
        time_values.append(Time)
        tre_values.append(diff_sum.item())
        rre_total += rre
        rte_total += rte
        time_total += Time
        tre_total += diff_sum.item()
        print(" num", num, f" TRE(mm): {diff_sum.item():.2f}", f" RRE(deg): {rre:.2f}, RTE(mm): {rte:.2f}, Time(s): {Time:.2f}")
        TRElist.append(diff_sum)
        RRElist.append(rre)
        RTElist.append(rte)
        Timelist.append(Time)

    TREmean = np.mean(TRElist)
    RREmean = np.mean(RRElist)
    RTEmean = np.mean(RTElist)
    Timemean = np.mean(Timelist)
    TREstd = np.std(TRElist)
    RREstd = np.std(RRElist)
    RTEstd = np.std(RTElist)
    Timestd = np.std(Timelist)

    print(f"Average TRE(mm): {TREmean:.2f}±{TREstd:.2f}")
    print(f"Average RRE(deg): {RREmean:.2f}±{RREstd:.2f}")
    print(f"Average RTE(mm): {RTEmean:.2f}±{RTEstd:.2f}")
    print(f"Average Time(s): {Timemean:.2f}±{Timestd:.2f}")
