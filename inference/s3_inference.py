import datetime
import sys
import os

import click
import numpy as np
from torch.utils.data import DataLoader
import torch

# Adds the repo root to sys.path
repo_root = os.path.abspath(os.path.join(os.getcwd(), '..'))
sys.path.append(repo_root)
sys.path.append(f'{repo_root}/pointnet2')
sys.path.append(f'{repo_root}/utils')
sys.path.append(f'{repo_root}/models')
sys.path.append(f'{repo_root}/dataset')

from graspnetAPI.graspnet_eval import GraspGroup
from models.graspnet import GraspNet, pred_decode
from dataset.s3_inference_dataset import FingerGraspDataset
from dataset.graspnet_dataset import minkowski_collate_fn


@click.command()
@click.option("-d", "--dataset-uri", type=str, default="s3://covariant-datasets-prod/depth_test_1741935319", help="The S3 URI of the dataset to infer on")
@click.option("-c", "--checkpoint-path", type=str, default="/home/ubuntu/model_ckpt/graspness/minkuresunet_realsense.tar", help="The path to the checkpoint file")
@click.option("-s", "--save-path", type=str, default="/home/ubuntu/test_results/graspness/grasp_points", help="The path to save the grasp points")
def main(dataset_uri: str, checkpoint_path: str, save_path: str):
    test_dataset = FingerGraspDataset(s3_uri=dataset_uri, voxel_size=0.005, num_points=15000)
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False,
                                 num_workers=0, worker_init_fn=my_worker_init_fn, collate_fn=minkowski_collate_fn)
    
    net = GraspNet(seed_feat_dim=512, is_training=False)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net.to(device)
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path)
    net.load_state_dict(checkpoint['model_state_dict'])
    start_epoch = checkpoint['epoch']
    print("-> loaded checkpoint %s (epoch: %d)" % (checkpoint_path, start_epoch))

    net.eval()

    save_path = os.path.join(save_path, f'{datetime.datetime.now().strftime("%m_%d_%H_%M")}')
    if not os.path.exists(save_path):
        os.makedirs(save_path)
        
    
    for batch_idx, batch_data in enumerate(test_dataloader):
        for key in batch_data:
            batch_data[key]=batch_data[key].to(device)
        
        with torch.no_grad():
            end_points = net(batch_data)
            grasp_preds = pred_decode(end_points)

        preds = grasp_preds[0].detach().cpu().numpy()

        gg = GraspGroup(preds)
        grasp_save_path = os.path.join(save_path, f'{batch_idx}.npy')
        gg.save_npy(grasp_save_path)

        print(f"Successfully saved grasp points for scene {batch_idx}")


def my_worker_init_fn(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)
    pass

if __name__ == "__main__":
    main()

