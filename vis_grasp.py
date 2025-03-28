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
    #vis = o3d.visualization.Visualizer()
    #vis.create_window(visible=False, width=1920, height=1080)
    
    # Set up the renderer
    #render_option = o3d.visualization.RenderOption()
    #render_option.background_color = np.asarray([0, 0, 0])  # Black background
    #render_option.point_size = 1.0
    #render_option.light_on = True
    #vis.get_render_option().load_from_json(render_option.to_json())
    
    # Add each geometry individually
    #for geometry in geometries:
    #    vis.add_geometry(geometry)
    
    # Render and capture
    #vis.poll_events()
    #vis.update_renderer()
    #vis.capture_screen_image(f"/home/ubuntu/logs/images/scene_{sceneId}_{camera}_{annId}.png")
    #vis.destroy_window()
   # Create a red sphere
    mesh = o3d.geometry.TriangleMesh.create_sphere(radius=1.0)
    mesh.compute_vertex_normals()
    mesh.paint_uniform_color([1, 0, 0])  # Red color

    # Set up material (optional, ensures visibility)
    material = o3d.visualization.rendering.MaterialRecord()
    material.albedo = [1.0, 0.0, 0.0, 1.0]  # RGBA for red

    # Initialize off-screen renderer
    renderer = o3d.visualization.rendering.OffscreenRenderer(width=800, height=600)

    # Set up scene
    renderer.scene.add_geometry("sphere", mesh, material)
    renderer.scene.set_background([0.0, 0.0, 0.0, 1.0])  # Black background

    # Add lighting
    renderer.scene.scene.set_sun_light(
        direction=[0, 0, -1],  # Light direction
        color=[1, 1, 1],       # White light
        intensity=100000       # Bright enough to illuminate
    )

    # Configure camera
    renderer.scene.camera.look_at(
        center=[0, 0, 0],      # Look at sphere’s center
        eye=[0, 0, 2],         # Camera position
        up=[0, 1, 0]           # Up direction
    )

    # Render and save
    output_path = "/home/ubuntu/logs/images/test.png"
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
