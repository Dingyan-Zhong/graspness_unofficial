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

    # Initialize off-screen renderer
    renderer = o3d.visualization.rendering.OffscreenRenderer(width=800, height=600)

    # Set up scene
    #renderer.scene.add_geometry("sphere", mesh, material)
    geometries = gg.to_open3d_geometry_list()
    for i in range(len(geometries)):
        renderer.scene.add_geometry("grasp_" + str(i), geometries[i], o3d.visualization.rendering.MaterialRecord())
    renderer.scene.add_geometry("scene", scenePCD, o3d.visualization.rendering.MaterialRecord())
    renderer.scene.set_background([1.0, 1.0, 1.0, 1.0])  # Black background

    # Add lighting
    renderer.scene.scene.set_sun_light(
        direction=[0, 0, -1],  # Light direction
        color=[1, 1, 1],       # White light
        intensity=100000       # Bright enough to illuminate
    )

    # --- Configure Camera Based on Point Cloud Only ---
    pcd_center = scenePCD.get_center()  # Center of the point cloud
    pcd_extent = max(scenePCD.get_max_bound() - scenePCD.get_min_bound())  # Rough size estimate
    renderer.scene.camera.look_at(
        center=pcd_center,               # Look at point cloud’s center
        eye=pcd_center + [0, 0, pcd_extent * 2],  # Camera distance based on point cloud size
        up=[0, 1, 0]                     # Up direction
    )

    # Render and save
    output_path = "/home/ubuntu/logs/images/scene_" + str(sceneId) + "_" + camera + "_" + str(annId) + ".png"
    img = renderer.render_to_image()
    o3d.io.write_image(output_path, img)
    print("Image saved:", output_path)

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
