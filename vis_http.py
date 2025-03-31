import numpy as np
import os
import sys
import argparse
from flask import Flask, render_template, jsonify
import json
from typing import List, Tuple, Optional
import open3d as o3d
import graspnetAPI
from graspnetAPI import grasp, graspnet

def create_mesh_box(width, height, depth):
    vertices = np.array([
        [0, 0, 0],
        [width, 0, 0],
        [width, height, 0],
        [0, height, 0],
        [0, 0, depth],
        [width, 0, depth],
        [width, height, depth],
        [0, height, depth]
    ])
    triangles = np.array([
        [0, 1, 2], [0, 2, 3],  # front
        [4, 5, 6], [4, 6, 7],  # back
        [0, 1, 5], [0, 5, 4],  # bottom
        [2, 3, 7], [2, 7, 6],  # top
        [0, 3, 7], [0, 7, 4],  # left
        [1, 2, 6], [1, 6, 5]   # right
    ])
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(vertices)
    mesh.triangles = o3d.utility.Vector3iVector(triangles)
    return mesh

def grasp_to_open3d_gripper(grasp_data, color=None):
    # grasp_data is a 1D numpy array with 17 elements:
    # [score, width, height, depth, rotation_matrix(9), translation(3), object_id]
    score, width, height, depth = grasp_data[0:4]
    R = grasp_data[4:13].reshape(3, 3)
    center = grasp_data[13:16]
    # object_id is grasp_data[16], not used here

    x, y, z = center
    height_mesh = 0.004
    finger_width = 0.004
    tail_length = 0.04
    depth_base = 0.02

    if color is not None:
        color_r, color_g, color_b = color
    else:
        color_r = score  # red for high score
        color_g = 0
        color_b = 1 - score  # blue for low score

    left = create_mesh_box(depth + depth_base + finger_width, finger_width, height_mesh)
    right = create_mesh_box(depth + depth_base + finger_width, finger_width, height_mesh)
    bottom = create_mesh_box(finger_width, width, height_mesh)
    tail = create_mesh_box(tail_length, finger_width, height_mesh)

    left_points = np.array(left.vertices)
    left_triangles = np.array(left.triangles)
    left_points[:, 0] -= depth_base + finger_width
    left_points[:, 1] -= width / 2 + finger_width
    left_points[:, 2] -= height_mesh / 2

    right_points = np.array(right.vertices)
    right_triangles = np.array(right.triangles) + 8
    right_points[:, 0] -= depth_base + finger_width
    right_points[:, 1] += width / 2
    right_points[:, 2] -= height_mesh / 2

    bottom_points = np.array(bottom.vertices)
    bottom_triangles = np.array(bottom.triangles) + 16
    bottom_points[:, 0] -= finger_width + depth_base
    bottom_points[:, 1] -= width / 2
    bottom_points[:, 2] -= height_mesh / 2

    tail_points = np.array(tail.vertices)
    tail_triangles = np.array(tail.triangles) + 24
    tail_points[:, 0] -= tail_length + finger_width + depth_base
    tail_points[:, 1] -= finger_width / 2
    tail_points[:, 2] -= height_mesh / 2

    vertices = np.concatenate([left_points, right_points, bottom_points, tail_points], axis=0)
    vertices = np.dot(R, vertices.T).T + center
    triangles = np.concatenate([left_triangles, right_triangles, bottom_triangles, tail_triangles], axis=0)
    colors = np.array([[color_r, color_g, color_b] for _ in range(len(vertices))])

    gripper = o3d.geometry.TriangleMesh()
    gripper.vertices = o3d.utility.Vector3dVector(vertices)
    gripper.triangles = o3d.utility.Vector3iVector(triangles)
    gripper.vertex_colors = o3d.utility.Vector3dVector(colors)
    return gripper

def open3d_mesh_to_threejs_json(mesh):
    vertices = np.array(mesh.vertices)
    triangles = np.array(mesh.triangles)
    colors = np.array(mesh.vertex_colors)
    # Three.js expects flat arrays
    vertices_flat = vertices.flatten().tolist()
    triangles_flat = triangles.flatten().tolist()
    colors_flat = colors.flatten().tolist()
    return {
        'vertices': vertices_flat,
        'triangles': triangles_flat,
        'colors': colors_flat
    }

app = Flask(__name__)

# Global variables to store visualization data
point_cloud_data = None
point_cloud_colors = None
grasp_points_data = None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/data')
def get_data():
    global point_cloud_data, point_cloud_colors, grasp_points_data
    grasp_meshes = []
    if grasp_points_data is not None:
        for grasp in grasp_points_data:
            gripper_mesh = grasp_to_open3d_gripper(grasp)
            mesh_json = open3d_mesh_to_threejs_json(gripper_mesh)
            grasp_meshes.append(mesh_json)
    return jsonify({
        'point_cloud': point_cloud_data.tolist() if point_cloud_data is not None else None,
        'point_cloud_colors': point_cloud_colors.tolist() if point_cloud_colors is not None else None,
        'grasp_meshes': grasp_meshes
    })

def load_data(root_dir: str, scene_id: int, camera: str, ann_id: int, grasp_points_path: Optional[str] = None):
    """Load scene point cloud and grasp data using graspnetAPI."""
    global point_cloud_data, point_cloud_colors, grasp_points_data
    gnet = graspnet.GraspNet(root=root_dir, camera=camera, split="test")
    points, colors = gnet.loadScenePointCloud(sceneId=scene_id, camera=camera, annId=ann_id, format='numpy')
    point_cloud_data = points
    point_cloud_colors = colors
    if grasp_points_path:
        grasp_points_data = np.load(grasp_points_path)

def main():
    parser = argparse.ArgumentParser(description='Web-based visualization of point cloud and grasp points')
    parser.add_argument('--scene_id', type=int, required=True, help='Scene ID')
    parser.add_argument('--camera', type=str, required=True, help='Camera type (e.g., realsense)')
    parser.add_argument('--ann_id', type=int, required=True, help='Annotation ID')
    parser.add_argument('--grasp_points', type=str, help='Path to grasp points data (numpy array)')
    parser.add_argument('--host', type=str, default='0.0.0.0', help='Host to run the server on')
    parser.add_argument('--port', type=int, default=5000, help='Port to run the server on')
    parser.add_argument('--root', type=str, default='/path/to/graspnet', help='Path to graspnet root')
    args = parser.parse_args()
    
    # Load data
    load_data(args.root, args.scene_id, args.camera, args.ann_id, args.grasp_points)
    
    # Create templates directory if it doesn't exist
    os.makedirs('templates', exist_ok=True)
    
    # Create the HTML template
    with open('templates/index.html', 'w') as f:
        f.write('''
<!DOCTYPE html>
<html>
<head>
    <title>Point Cloud and Grasp Visualization</title>
    <style>
        body { margin: 0; }
        canvas { width: 100%; height: 100% }
    </style>
</head>
<body>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/controls/OrbitControls.js"></script>
    <script>
        let scene, camera, renderer, pointCloud, controls;
        let graspMeshes = [];
        
        function init() {
            // Create scene
            scene = new THREE.Scene();
            scene.background = new THREE.Color(0x1a1a1a);
            
            // Create camera
            camera = new THREE.PerspectiveCamera(75, window.innerWidth / window.innerHeight, 0.1, 1000);
            camera.position.z = 5;
            
            // Create renderer
            renderer = new THREE.WebGLRenderer({ antialias: true });
            renderer.setSize(window.innerWidth, window.innerHeight);
            document.body.appendChild(renderer.domElement);
            
            // Add lights
            const ambientLight = new THREE.AmbientLight(0xffffff, 0.5);
            scene.add(ambientLight);
            
            const directionalLight = new THREE.DirectionalLight(0xffffff, 0.5);
            directionalLight.position.set(0, 1, 0);
            scene.add(directionalLight);
            
            controls = new THREE.OrbitControls(camera, renderer.domElement);
            controls.enableDamping = true;
            controls.dampingFactor = 0.05;
            
            // Load data
            fetch('/data')
                .then(response => response.json())
                .then(data => {
                    if (data.point_cloud) {
                        // Create point cloud
                        const geometry = new THREE.BufferGeometry();
                        const positions = new Float32Array(data.point_cloud.flat());
                        geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
                        
                        let material;
                        if (data.point_cloud_colors) {
                            const colors = new Float32Array(data.point_cloud_colors.flat());
                            geometry.setAttribute('color', new THREE.BufferAttribute(colors, 3));
                            material = new THREE.PointsMaterial({
                                size: 0.02,
                                vertexColors: true,
                                sizeAttenuation: true
                            });
                        } else {
                            material = new THREE.PointsMaterial({
                                size: 0.02,
                                color: 0x808080,
                                sizeAttenuation: true
                            });
                        }
                        
                        pointCloud = new THREE.Points(geometry, material);
                        scene.add(pointCloud);
                    }
                    
                    if (data.grasp_meshes) {
                        data.grasp_meshes.forEach(meshData => {
                            const geometry = new THREE.BufferGeometry();
                            const vertices = new Float32Array(meshData.vertices);
                            const indices = new Uint32Array(meshData.triangles);
                            const colors = new Float32Array(meshData.colors);

                            geometry.setAttribute('position', new THREE.BufferAttribute(vertices, 3));
                            geometry.setIndex(new THREE.BufferAttribute(indices, 1));
                            geometry.setAttribute('color', new THREE.BufferAttribute(colors, 3));

                            const material = new THREE.MeshPhongMaterial({
                                vertexColors: true,
                                side: THREE.DoubleSide
                            });

                            const mesh = new THREE.Mesh(geometry, material);
                            scene.add(mesh);
                            graspMeshes.push(mesh);
                        });
                    }
                });
            
            // Handle window resize
            window.addEventListener('resize', onWindowResize, false);
        }
        
        function onWindowResize() {
            camera.aspect = window.innerWidth / window.innerHeight;
            camera.updateProjectionMatrix();
            renderer.setSize(window.innerWidth, window.innerHeight);
        }
        
        function animate() {
            requestAnimationFrame(animate);
            controls.update();
            renderer.render(scene, camera);
        }
        
        init();
        animate();
    </script>
</body>
</html>
        ''')
    
    # Run the Flask app
    app.run(host=args.host, port=args.port)

if __name__ == "__main__":
    main()
