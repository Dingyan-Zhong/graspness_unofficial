import numpy as np
import graspnetAPI
from graspnetAPI import grasp, graspnet
import open3d as o3d

def show_predicted_grasp_6d(gnet, sceneId, camera, annId, grasps, show_object=False):
    geometries = []
    scenePCD = gnet.loadScenePointCloud(sceneId = sceneId, camera = camera, annId = annId, align = False)
    #geometries.append(scenePCD)
    gg = grasps.nms()
    gg = gg.sort_by_score()
    if gg.__len__() > 30:
        gg = gg[:30]

    # Set up scene
    #renderer.scene.add_geometry("sphere", mesh, material)
    geometries = gg.to_open3d_geometry_list()

    # --- Launch Web Visualizer ---
    o3d.visualization.webrtc_server.enable_webrtc()
    o3d.visualization.draw(
        geometries=[scenePCD] + geometries,
        window_name="3D Scene",
        width=1920,
        height=1080,
        lookat=scenePCD.get_center().tolist(),
        eye=(scenePCD.get_center() + [0.5, 0.3, 1.2]).tolist(),
        up=[0, 1, 0]
    )

def create_grasp_group(grasp_group_path):
    grasp_group = np.load(grasp_group_path)
    #grasp_group = graspnet.GraspGroup(grasp_group)
    gnet = graspnet.GraspGroup()
    for grasp in grasp_group:
        gnet.grasp_group_array = np.concatenate((gnet.grasp_group_array, grasp.reshape(1,-1)))
    return gnet

def show_predicted_grasp_6d_from_file(gnet, grasp_group_path, sceneId, camera, annId):
    grasp_group = create_grasp_group(grasp_group_path)
    show_predicted_grasp_6d(gnet, sceneId, camera, annId, grasp_group)

if __name__ == "__main__":
    grasp_group_path = "/home/ubuntu/logs/scene_0100/realsense/0000.npy"
    sceneId = 100
    camera = "realsense"
    annId = 0
    gnet = graspnet.GraspNet(root="/home/ubuntu/graspnet", camera="realsense", split="test")
    show_predicted_grasp_6d_from_file(gnet, grasp_group_path, sceneId, camera, annId)
