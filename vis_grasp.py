import numpy as np
import graspnetAPI
from graspnetAPI import grasp, graspnet
import open3d as o3d

def show_predicted_grasp_6d(gnet, sceneId, camera, annId, grasps, show_object=False):
    geometries = []
    scenePCD = gnet.loadScenePointCloud(sceneId = sceneId, camera = camera, annId = annId, align = False)
    geometries.append(scenePCD)
    gg = grasps.nms()
    gg = gg.sort_by_score()
    if gg.__len__() > 30:
        gg = gg[:30]
    geometries += gg.to_open3d_geometry_list()
    
    # Create a new visualizer with headless rendering
    vis = o3d.visualization.Visualizer()
    vis.create_window(visible=False, width=1920, height=1080)
    
    # Set up the renderer
    render_option = vis.get_render_option()
    render_option.background_color = np.asarray([0, 0, 0])  # Black background
    render_option.point_size = 1.0
    render_option.light_on = True
    
    # Add each geometry individually
    for geometry in geometries:
        vis.add_geometry(geometry)
    
    # Set the camera view
    ctr = vis.get_view_control()
    ctr.set_zoom(0.8)
    
    # Render and capture
    vis.poll_events()
    vis.update_renderer()
    vis.capture_screen_image(f"/home/ubuntu/logs/images/scene_{sceneId}_{camera}_{annId}.png")
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
