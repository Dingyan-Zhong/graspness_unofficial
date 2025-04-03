import os
import pandas as pd
import numpy as np
import json
import boto3

import torch
from torch.utils.data import Dataset

import cv2
from io import BytesIO
from utils.data_utils import CameraInfo, create_point_cloud_from_depth_image
from zstandard import ZstdDecompressor

S3_PREFIX = "s3://"

class S3InferenceDataset(Dataset):
    def __init__(self, s3_uri: str, voxel_size: float = 0.05, num_points: int = 10000):
        self.s3_client = boto3.client("s3")
        self.data = pd.read_parquet(s3_uri)
        self.voxel_size = voxel_size
        self.num_points = num_points

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        datum = self.data.iloc[idx]

        camera_ids = datum["camera_ids"]

        depth_maps_path = json.loads(datum["depth_maps_path"])
        reference_camera_id = list(depth_maps_path.keys())[0]
        depth_maps_path = depth_maps_path[reference_camera_id]

        # Load depth map
        depth_map = load_tensor_s3(depth_map, self.s3_client)

        # Load 2d image
        images_path = json.loads(datum["images_path"])
        img = images_path[reference_camera_id]
        img = deserialize_and_download_image(img, self.s3_client)

        # Load camera intrinsic
        camera_intrinsics = json.loads(datum["camera_intrinsic_path"])
        camera_intrinsics = load_tensor_s3(camera_intrinsics, self.s3_client)
        reference_camera_intrinsics = camera_intrinsics[np.where(camera_ids == reference_camera_id)[0][0]]

        camera = CameraInfo(
            width=depth_map.shape[1],
            height=depth_map.shape[0],
            fx=reference_camera_intrinsics[0][0],
            fy=reference_camera_intrinsics[1][1],
            cx=reference_camera_intrinsics[0][2],
            cy=reference_camera_intrinsics[1][2],
            scale=1.0
        )
        # Compute 3d point cloud
        # TODO: really use meters?
        cloud = create_point_cloud_from_depth_image(depth_map, camera, organized=True)
    
        # 3. Get valid points (just remove zero depth)
        depth_mask = (depth_map > 0)
        cloud_masked = cloud[depth_mask]

        # 4. Sample points
        if len(cloud_masked) >= self.num_points:
            idxs = np.random.choice(len(cloud_masked), self.num_points, replace=False)
        else:
            idxs1 = np.arange(len(cloud_masked))
            idxs2 = np.random.choice(len(cloud_masked), self.num_points - len(cloud_masked), replace=True)
            idxs = np.concatenate([idxs1, idxs2], axis=0)
        cloud_sampled = cloud_masked[idxs]

        # 5. Prepare model input
        model_input = {
            'point_clouds': cloud_sampled.astype(np.float32),
            'coors': cloud_sampled.astype(np.float32) / self.voxel_size,
            'feats': np.ones_like(cloud_sampled).astype(np.float32)
        }
    
        return model_input 
        



def deserialize_and_download_image(
    s3_uri, client, bit_depth: int = 8, dtype: torch.dtype = torch.float32
) -> torch.Tensor:
    """Shared utility for DeserializedObjectView and DeserializedImage.

    Look at those class docstrings for more information.
    """
    bucket, key = get_s3_bucket_key(s3_uri)
    image_bytes = client.get_object(Bucket=bucket, Key=key)["Body"].read()
    if bit_depth == 8:
        image_np = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
        image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
    elif bit_depth > 8 and bit_depth <= 16:
        # note that torch half starts losing precision for bit_depth > 11; it becomes a choice for the user to
        # tradeoff loading speed vs precision. For bit_depth=12, the max error is 1px (out of 4096 slots).
        if dtype not in {torch.float, torch.half}:
            raise ValueError(f"dtype must be torch.float or torch.half if bit_depth > 8, not {dtype}")
        image_np = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_ANYDEPTH | cv2.IMREAD_COLOR)
        image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
        # have to convert to float16 or float32 first, since np.uint16 is not supported by pytorch
        dtype_np = np.float16 if dtype == torch.half else np.float32
        image_np = image_np.astype(dtype_np)
    else:
        raise ValueError(f"bit_depth must be in the range [8, 16], not {bit_depth}!")

    return image_np

def load_tensor_s3(tensor_uri, s3_client):
    bucket, key = get_s3_bucket_key(tensor_uri)
    tensor_bytes = s3_client.get_object(Bucket=bucket, Key=key)["Body"].read()
    tensor_bytes = ZstdDecompressor().decompress(tensor_bytes)
    tensor_np =  np.load(BytesIO(tensor_bytes), allow_pickle=bool(os.environ.get("COV_ALLOW_UNPICKLE", "0")))
    return tensor_np
    

def get_s3_bucket_key(uri):
    bucket, prefix = uri[len(S3_PREFIX) :].split("/", maxsplit=1)
    return bucket, prefix