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

    vis = o3d.visualization.Visualizer()
    vis.create_window(width=1920, height=1080)

    # Set up scene
    #renderer.scene.add_geometry("sphere", mesh, material)
    geometries = gg.to_open3d_geometry_list()
    for i in range(len(geometries)):
        vis.add_geometry(geometries[i])
    vis.add_geometry(scenePCD)

    render_option = vis.get_render_option()
    render_option.background_color = np.array([1.0, 1.0, 1.0])  # White background (RGB)

    ctr = vis.get_view_control()
    pcd_center = scenePCD.get_center()
    pcd_extent = max(scenePCD.get_max_bound() - scenePCD.get_min_bound())
    ctr.set_lookat(pcd_center)
    ctr.set_front([0, 0, -1])
    ctr.set_up([0, 1, 0])
    ctr.set_zoom(pcd_extent * 0.7 / 10)  # Adjust zoom (smaller = closer)

    # Run interactive loop
    vis.run()
    vis.destroy_window()

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
