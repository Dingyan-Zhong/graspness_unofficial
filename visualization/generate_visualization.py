import numpy as np
import os
import sys
import argparse
import json
import traceback
from io import BytesIO

# --- Pythreejs and Embedding Imports ---
import pythreejs as three
from ipywidgets.embed import embed_minimal_html

# --- S3 and Data Loading Imports ---
import boto3
import pandas as pd
import cv2 # OpenCV for image loading
from zstandard import ZstdDecompressor

# --- Constants ---
S3_PREFIX = "s3://"

# Add repository root to path if necessary (adjust based on your structure)
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if repo_root not in sys.path:
    sys.path.append(repo_root)

# --- CameraInfo Class (Copied from utils/data_utils.py or similar) ---
class CameraInfo():
    """ Camera intrinsics structure. """
    def __init__(self, width, height, fx, fy, cx, cy, scale):
        self.width = width
        self.height = height
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy
        self.scale = scale # Renamed from factor_depth for consistency

# --- Point Cloud Generation (Copied from utils/data_utils.py or similar) ---
def create_point_cloud_from_depth_image(depth, camera: CameraInfo):
    """ Generate point cloud using depth image only. """
    # Ensure depth is float32 for calculations
    if depth.dtype != np.float32:
        depth = depth.astype(np.float32)

    if not (depth.shape[0] == camera.height and depth.shape[1] == camera.width):
        raise ValueError("Depth map dimensions do not match camera height/width.")

    xmap = np.arange(camera.width)
    ymap = np.arange(camera.height)
    xmap, ymap = np.meshgrid(xmap, ymap)

    points_z = depth / camera.scale # Use camera.scale (factor_depth)
    points_x = (xmap - camera.cx) * points_z / camera.fx
    points_y = (ymap - camera.cy) * points_z / camera.fy
    cloud = np.stack([points_x, points_y, points_z], axis=-1)
    return cloud # Return the organized (H, W, 3) cloud

# --- S3 Utility Functions (Adapted from dataset/s3_inference_dataset.py) ---
def get_s3_bucket_key(uri):
    """Extracts S3 bucket and key from URI."""
    if not uri.startswith(S3_PREFIX):
        raise ValueError(f"Invalid S3 URI: {uri}")
    bucket, key = uri[len(S3_PREFIX) :].split("/", maxsplit=1)
    return bucket, key

def load_tensor_s3(tensor_uri, s3_client):
    """Loads a compressed numpy tensor from S3."""
    print(f"    Loading tensor: {tensor_uri}")
    bucket, key = get_s3_bucket_key(tensor_uri)
    try:
        response = s3_client.get_object(Bucket=bucket, Key=key)
        tensor_bytes = response["Body"].read()
        # Check if compressed (simple check for zstd header bytes)
        if tensor_bytes.startswith(b'\x28\xb5\x2f\xfd'):
            tensor_bytes = ZstdDecompressor().decompress(tensor_bytes)
        # Use allow_pickle=True cautiously if necessary, otherwise False
        allow_pickle = bool(os.environ.get("COV_ALLOW_UNPICKLE", "0"))
        tensor_np = np.load(BytesIO(tensor_bytes), allow_pickle=allow_pickle)
        return tensor_np
    except Exception as e:
        print(f"    Error loading tensor {tensor_uri}: {e}")
        raise # Re-raise the exception to halt processing for this item

def deserialize_and_download_image(s3_uri, client):
    """Downloads and decodes an image from S3 (assumes standard formats like PNG/JPG)."""
    print(f"    Loading image: {s3_uri}")
    bucket, key = get_s3_bucket_key(s3_uri)
    try:
        response = client.get_object(Bucket=bucket, Key=key)
        image_bytes = response["Body"].read()
        image_np = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
        if image_np is None:
            raise ValueError("cv2.imdecode returned None. Invalid image format or data.")
        # Convert from BGR (OpenCV default) to RGB
        image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
        return image_np
    except Exception as e:
        print(f"    Error loading image {s3_uri}: {e}")
        raise # Re-raise

# --- Helper Functions for Geometry and NMS ---

def create_mesh_box_vertices_indices(width, height, depth):
    """Generates vertices and indices for a box mesh."""
    vertices = np.array([
        [0, 0, 0], [width, 0, 0], [width, height, 0], [0, height, 0],
        [0, 0, depth], [width, 0, depth], [width, height, depth], [0, height, depth]
    ], dtype=np.float32)
    indices = np.array([
        0, 1, 2, 0, 2, 3,  # front
        4, 5, 6, 4, 6, 7,  # back
        0, 1, 5, 0, 5, 4,  # bottom
        2, 3, 7, 2, 7, 6,  # top
        0, 3, 7, 0, 7, 4,  # left
        1, 2, 6, 1, 6, 5   # right
    ], dtype=np.uint32)
    return vertices, indices

def get_gripper_mesh_data(grasp_data):
    """
    Generates vertices, indices, and colors for a gripper mesh based on grasp data.
    Args:
        grasp_data (np.ndarray): 1D numpy array with 17 elements:
            [score, width, height, depth, rotation_matrix(9), translation(3), object_id]
    Returns:
        dict: Contains 'vertices', 'indices', 'colors' for the gripper mesh.
    """
    score, width, height, depth = grasp_data[0:4]
    R = grasp_data[4:13].reshape(3, 3)
    center = grasp_data[13:16]

    # Gripper geometry parameters
    height_mesh = 0.004
    finger_width = 0.004
    tail_length = 0.04
    depth_base = 0.02

    # Color based on score (Red, varying intensity)
    intensity = 0.5 + 0.5 * score
    color_val = [intensity, 0.0, 0.0]

    # Create components (vertices and indices)
    left_v, left_i = create_mesh_box_vertices_indices(depth + depth_base + finger_width, finger_width, height_mesh)
    right_v, right_i = create_mesh_box_vertices_indices(depth + depth_base + finger_width, finger_width, height_mesh)
    bottom_v, bottom_i = create_mesh_box_vertices_indices(finger_width, width, height_mesh)
    tail_v, tail_i = create_mesh_box_vertices_indices(tail_length, finger_width, height_mesh)

    # Transform components to be relative to the center (before rotation)
    left_v[:, 0] -= depth_base + finger_width
    left_v[:, 1] -= width / 2 + finger_width
    left_v[:, 2] -= height_mesh / 2

    right_v[:, 0] -= depth_base + finger_width
    right_v[:, 1] += width / 2
    right_v[:, 2] -= height_mesh / 2

    bottom_v[:, 0] -= finger_width + depth_base
    bottom_v[:, 1] -= width / 2
    bottom_v[:, 2] -= height_mesh / 2

    tail_v[:, 0] -= tail_length + finger_width + depth_base
    tail_v[:, 1] -= finger_width / 2
    tail_v[:, 2] -= height_mesh / 2

    # Combine vertices and indices, adjusting indices
    vertices = np.vstack([left_v, right_v, bottom_v, tail_v])
    indices = np.hstack([
        left_i,
        right_i + 8,   # Offset for right vertices
        bottom_i + 16, # Offset for bottom vertices
        tail_i + 24    # Offset for tail vertices
    ])

    # Apply rotation and translation
    vertices_transformed = np.dot(vertices, R.T) + center

    # Create colors array (one color per vertex)
    colors = np.array([color_val] * len(vertices_transformed), dtype=np.float32)

    # Return data suitable for pythreejs BufferGeometry attributes
    return {
        'vertices': vertices_transformed.flatten().tolist(), # Flatten for BufferAttribute
        'indices': indices.tolist(),
        'colors': colors.flatten().tolist() # Flatten for BufferAttribute
    }

def compute_iou(grasp1, grasp2):
    """Compute IoU between two grasps based on centers and rotations."""
    center1 = grasp1[13:16]
    center2 = grasp2[13:16]
    R1 = grasp1[4:13].reshape(3, 3)
    R2 = grasp2[4:13].reshape(3, 3)
    center_dist = np.linalg.norm(center1 - center2)
    if center_dist > 0.1: return 0.0
    R_diff = np.dot(R1, R2.T)
    trace = np.trace(R_diff)
    # Clip trace to avoid numerical issues with arccos
    trace = np.clip(trace, -1.0, 3.0)
    angle = np.arccos(np.clip((trace - 1) / 2, -1.0, 1.0))
    if angle > np.pi/4: return 0.0
    iou = (1 - center_dist/0.1) * (1 - angle/(np.pi/4))
    return max(0.0, iou)

def nms(grasps, iou_threshold=0.5, max_grasps=50):
    """Apply Non-Maximum Suppression to grasp points."""
    if grasps is None or len(grasps) == 0:
        return np.array([])
    grasps = np.asarray(grasps)
    if grasps.ndim == 1: grasps = grasps[np.newaxis, :]
    if grasps.shape[1] != 17:
         raise ValueError(f"Expected grasp data to have 17 elements, but got shape {grasps.shape}")

    scores = grasps[:, 0]
    indices = np.argsort(scores)[::-1]
    grasps_sorted = grasps[indices]
    kept_grasps = []
    while len(grasps_sorted) > 0 and len(kept_grasps) < max_grasps:
        current_grasp = grasps_sorted[0]
        kept_grasps.append(current_grasp)
        if len(grasps_sorted) == 1: break

        ious = np.array([compute_iou(current_grasp, g) for g in grasps_sorted[1:]])
        keep_indices = np.where(ious <= iou_threshold)[0]
        grasps_sorted = grasps_sorted[keep_indices + 1]

    return np.array(kept_grasps)

# --- Step 1: Data Loading and Processing Function (S3 Version) ---
def load_and_process_s3_scene(datum, s3_client, grasp_path, depth_scale, scene_key, max_grasps=50):
    """Loads scene data from S3 (using manifest row) and local grasps, processes for visualization."""
    print(f"Processing {scene_key}...")
    try:
        # 1. Extract S3 URIs from the manifest row (datum)
        # These column names are assumed based on s3_inference_dataset.py
        # Adjust column names if your Parquet schema is different!
        camera_ids = datum["camera_ids"]
       
        depth_maps_path = json.loads(datum["depth_maps_path"])
        reference_camera_id = list(depth_maps_path.keys())[0]
        depth_maps_path = depth_maps_path[reference_camera_id]

        images_path = json.loads(datum["images_path"])
        image_path = images_path[reference_camera_id]

        camera_intrinsics = datum["camera_intrinsics_path"]
        camera_intrinsics = load_tensor_s3(camera_intrinsics, s3_client)
        
        # 2. Load data from S3
        depth_map_np = load_tensor_s3(depth_maps_path, s3_client)
        rgb_image_np = deserialize_and_download_image(image_path, s3_client)
        reference_camera_intrinsics = camera_intrinsics[np.where(camera_ids == reference_camera_id)[0][0]]

        # 3. Create CameraInfo
        height, width = depth_map_np.shape
        fx, fy = reference_camera_intrinsics[0, 0], reference_camera_intrinsics[1, 1]
        cx, cy = reference_camera_intrinsics[0, 2], reference_camera_intrinsics[1, 2]
        camera = CameraInfo(width, height, fx, fy, cx, cy, depth_scale)

        # 4. Generate Point Cloud
        print(f"    Generating point cloud ({height}x{width})...")
        cloud_organized = create_point_cloud_from_depth_image(depth_map_np, camera)

        # 5. Filter points and get corresponding colors
        valid_depth_mask = depth_map_np > 0
        points = cloud_organized[valid_depth_mask]
        colors = rgb_image_np[valid_depth_mask]

        # Ensure colors are float32 and normalized [0, 1]
        if colors.dtype != np.float32 or colors.max() > 1.0:
            # Check max value before dividing
            if colors.max() > 1:
                colors = colors.astype(np.float32) / 255.0
            else:
                colors = colors.astype(np.float32)
        elif colors.max() > 1.0: # Handle cases where dtype is float but not normalized
             colors = colors / 255.0

        print(f"    Filtered to {len(points)} valid points.")

        # 6. Load local grasp data
        raw_grasps = None
        if grasp_path and os.path.exists(grasp_path):
            try:
                raw_grasps = np.load(grasp_path)
                print(f"    Loaded {len(raw_grasps)} raw grasps from {grasp_path}")
            except Exception as e:
                print(f"    Warning: Failed to load local grasp file {grasp_path}: {e}")
        else:
             print(f"    Warning: Local grasp file not found: {grasp_path}")

        # 7. Apply Non-Maximum Suppression (NMS)
        filtered_grasps = nms(raw_grasps, max_grasps=max_grasps)
        print(f"    Filtered to {len(filtered_grasps)} grasps after NMS.")

        # 8. Pre-calculate gripper mesh data
        gripper_meshes_data = [get_gripper_mesh_data(g) for g in filtered_grasps]

        # 9. Return processed data
        return scene_key, {
            'point_cloud_vertices': points.flatten().tolist(),
            'point_cloud_colors': colors.flatten().tolist(),
            'gripper_meshes': gripper_meshes_data
        }

    except Exception as e:
        print(f"Error processing {scene_key}: {e}")
        traceback.print_exc()
        return None

# --- Step 2: Generate Interactive Scene HTML ---
def create_interactive_scene_html(scene_data, scene_key, output_dir):
    """Creates a self-contained HTML file for a single scene using pythreejs."""
    print(f"  Generating HTML for {scene_key}...")
    try:
        # 1. Basic Scene Setup
        camera = three.PerspectiveCamera(position=[1.0, 1.0, 1.0], aspect=16/9, fov=60, near=0.01, far=10)
        # Try to center view based on point cloud bounds
        pc_vertices_np = np.array(scene_data['point_cloud_vertices']).reshape(-1, 3)
        if pc_vertices_np.size > 0:
            view_center = np.mean(pc_vertices_np, axis=0)
        else:
            view_center = np.array([0.0, 0.0, 0.0]) # Default center if no points

        if not np.any(np.isnan(view_center)):
             camera.lookAt(view_center.tolist()) # Point camera towards cloud center

        scene = three.Scene()
        scene.add(three.AmbientLight(color='#ffffff', intensity=1.0))
        scene.add(three.DirectionalLight(color='#ffffff', position=[1, 1, 1], intensity=1.0))
        scene.add(three.DirectionalLight(color='#ffffff', position=[-1, -1, -1], intensity=0.5))

        # 2. Point Cloud
        pc_vertices = np.array(scene_data['point_cloud_vertices'], dtype=np.float32)
        pc_colors = np.array(scene_data['point_cloud_colors'], dtype=np.float32)

        # Handle empty point cloud case
        if pc_vertices.size == 0:
            print("    Warning: Empty point cloud, creating dummy geometry.")
            pc_vertices = np.zeros((1, 3), dtype=np.float32)
            pc_colors = np.zeros((1, 3), dtype=np.float32)

        pc_geometry = three.BufferGeometry(attributes={
            'position': three.BufferAttribute(array=pc_vertices, itemSize=3),
            'color': three.BufferAttribute(array=pc_colors, itemSize=3)
        })
        pc_material = three.PointsMaterial(vertexColors='VertexColors', size=0.003, sizeAttenuation=True)
        point_cloud_mesh = three.Points(geometry=pc_geometry, material=pc_material)
        scene.add(point_cloud_mesh)

        # 3. Gripper Meshes
        gripper_material = three.MeshPhongMaterial(vertexColors='VertexColors', side='DoubleSide', shininess=50, flatShading=False)
        for mesh_data in scene_data['gripper_meshes']:
            gripper_vertices = np.array(mesh_data['vertices'], dtype=np.float32)
            gripper_indices = np.array(mesh_data['indices'], dtype=np.uint32)
            gripper_colors = np.array(mesh_data['colors'], dtype=np.float32)

            # Handle potential empty mesh data (though unlikely from get_gripper_mesh_data)
            if gripper_vertices.size == 0 or gripper_indices.size == 0 or gripper_colors.size == 0:
                print(f"    Warning: Skipping empty gripper mesh data for {scene_key}.")
                continue

            geom = three.BufferGeometry(attributes={
                'position': three.BufferAttribute(array=gripper_vertices, itemSize=3),
                'index': three.BufferAttribute(array=gripper_indices, itemSize=1),
                'color': three.BufferAttribute(array=gripper_colors, itemSize=3)
            })
            mesh = three.Mesh(geometry=geom, material=gripper_material, visible=True)
            scene.add(mesh)

        # 4. Renderer and Controls
        orbit_controls = three.OrbitControls(controlling=camera)
        if not np.any(np.isnan(view_center)):
             orbit_controls.target = view_center.tolist() # Set orbit center

        renderer = three.Renderer(camera=camera, scene=scene, controls=[orbit_controls],
                                  width=1024, height=768)

        # 5. Define output path and save HTML
        # Sanitize scene_key for filename
        safe_filename = "".join(c if c.isalnum() or c in ('_', '-') else '_' for c in scene_key)
        output_filename = f"{safe_filename}.html"
        scenes_subdir = os.path.join(output_dir, 'scenes')
        os.makedirs(scenes_subdir, exist_ok=True)
        output_path = os.path.join(scenes_subdir, output_filename)

        embed_minimal_html(output_path, views=[renderer], title=scene_key)
        print(f"    Saved interactive HTML to: {output_path}")

        # Return the relative path for the index
        return os.path.join('scenes', output_filename)

    except Exception as e:
        print(f"Error generating HTML for {scene_key}: {e}")
        traceback.print_exc()
        return None

# --- Step 3: Generate Index HTML ---
def generate_index_html(scene_links, output_dir):
    """Generates an index.html file linking to individual scene visualizations."""
    print(f"\nGenerating index HTML...")
    index_path = os.path.join(output_dir, 'index.html')

    # Basic HTML structure
    html_start = """
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>Grasp Visualization Index</title>
    <style>
        body { font-family: sans-serif; margin: 20px; }
        h1 { text-align: center; }
        ul { list-style-type: none; padding: 0; }
        li { margin: 10px 0; background-color: #f4f4f4; padding: 10px; border-radius: 5px; }
        a { text-decoration: none; color: #007bff; font-weight: bold; }
        a:hover { text-decoration: underline; }
    </style>
</head>
<body>
    <h1>Grasp Visualization Index</h1>
    <ul>
"""
    html_end = """
    </ul>
</body>
</html>
"""

    list_items = []
    # Sort links alphabetically by scene key for consistent order
    for scene_key in sorted(scene_links.keys()):
        relative_path = scene_links[scene_key]
        # Ensure forward slashes for web paths, even on Windows
        web_path = relative_path.replace('\\', '/')
        list_items.append(f'        <li><a href="{web_path}" target="_blank">{scene_key}</a></li>')

    final_html = html_start + "\n".join(list_items) + "\n" + html_end

    try:
        with open(index_path, 'w', encoding='utf-8') as f:
            f.write(final_html)
        print(f"Saved index HTML to: {index_path}")
    except Exception as e:
        print(f"Error writing index HTML: {e}")


# --- Main Execution Logic (S3 Version) ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate grasp visualization HTML files from S3 manifest and local grasps.')
    parser.add_argument('--dataset_uri', type=str, required=True, help='S3 URI of the Parquet manifest file.')
    parser.add_argument('--grasp_dir', type=str, required=True, help='Local directory containing grasp prediction .npy files (named like 0.npy, 1.npy, ...).')
    parser.add_argument('--depth_scale', type=float, default=1, help='Factor to divide raw depth values by to get meters (e.g., 1000.0 for mm).')
    parser.add_argument('--max_grasps', type=int, default=10, help='Maximum number of grasps to process per scene after NMS.')
    parser.add_argument('--output_dir', type=str, default='visualization_output', help='Directory to save the output HTML files.')

    args = parser.parse_args()

    # Ensure output directory exists
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'scenes'), exist_ok=True) # Ensure scenes subdir exists

    # Initialize Boto3 S3 client
    print("Initializing S3 client...")
    try:
        s3_client = boto3.client("s3")
        # Optional: Add a check here to list buckets or test credentials if needed
        # Example: s3_client.list_buckets()
        print("S3 client initialized successfully.")
    except Exception as e:
        print(f"Error initializing Boto3 S3 client: {e}")
        print("Please ensure your AWS credentials and region are configured correctly (e.g., via environment variables or ~/.aws/credentials).")
        sys.exit(1)

    # Load Parquet manifest from S3
    print(f"Loading dataset manifest from: {args.dataset_uri}")
    manifest_local_path = "temp_manifest.parquet"
    try:
        # Assuming access to s3
        df = pd.read_parquet(args.dataset_uri)
        print(f"Loaded manifest with {len(df)} entries.")
    except Exception as e:
        print(f"Error loading Parquet manifest from {args.dataset_uri}: {e}")
        sys.exit(1)
    finally:
        # Clean up temporary file if it exists
        if os.path.exists(manifest_local_path):
            try:
                os.remove(manifest_local_path)
            except Exception as e_rem:
                 print(f"Warning: Failed to remove temporary manifest file {manifest_local_path}: {e_rem}")

    # Dictionary to store generated HTML paths
    scene_html_paths = {}

    print("\nStarting data processing and HTML generation...")
    # Loop through rows in the DataFrame manifest
    for idx in range(len(df)):
        datum = df.iloc[idx]
        # Construct expected local grasp file path
        grasp_filename = f"{idx}_04_15_09_56.npy" # Assumes naming convention 0.npy, 1.npy ...
        grasp_path = os.path.join(args.grasp_dir, grasp_filename)

        # Construct a scene key
        # Use index as key, or try to find more descriptive fields if available
        # Example: Prefer a more descriptive key if columns exist
        scene_name = datum.get('scene_name', f'scene_index') # Get scene name or default
        item_id = datum.get('item_id', idx) # Get item ID or use index
        scene_key = f"{scene_name}_{item_id}"

        print(f"\n--- Processing Entry {idx} ({scene_key}) ---")

        if not os.path.exists(grasp_path):
            print(f"  Warning: Local grasp file not found at {grasp_path}. Skipping {scene_key}.")
            continue

        # --- Step 1 --- #
        processed_data = load_and_process_s3_scene(
            datum, s3_client, grasp_path, args.depth_scale, scene_key, args.max_grasps
        )

        if processed_data:
            scene_key_returned, data_dict = processed_data # scene_key_returned should match scene_key
            print(f"  Successfully processed data for {scene_key}.")

            # --- Step 2 --- #
            relative_html_path = create_interactive_scene_html(
                data_dict, scene_key, args.output_dir
            )
            if relative_html_path:
                scene_html_paths[scene_key] = relative_html_path
            else:
                 print(f"  Failed to generate HTML for {scene_key}.")
        else:
            print(f"  Failed to process S3 data for {scene_key}.")

    # --- Step 3 --- #
    if scene_html_paths:
        generate_index_html(scene_html_paths, args.output_dir)
    else:
        print("\nNo scene HTML files were generated, skipping index.html creation.")

    print(f"\nFinished generation. Output files are in: {os.path.abspath(args.output_dir)}")

    # --- Final Instructions --- #
    print(f"\nTo view the visualization:")
    print(f"1. Navigate to the output directory:")
    print(f"   cd {os.path.abspath(args.output_dir)}")
    print(f"2. Start a simple HTTP server (requires Python 3):")
    print(f"   python -m http.server 8000")
    print(f"3. Open your web browser and go to: http://localhost:8000/")
    print("(The index.html file should load automatically)") 