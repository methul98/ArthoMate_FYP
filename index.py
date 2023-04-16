import base64
import io
import pickle
import cv2
import numpy as np
import open3d as o3d
import torch
import trimesh
from PIL import Image
from flask import Flask, render_template, request
from jinja2 import Environment
from transformers import GLPNImageProcessor, GLPNForDepthEstimation
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
import xgboost as xgb

def predict_depth(image):
    feature_extractor = GLPNImageProcessor.from_pretrained("vinvino02/glpn-nyu")
    model = GLPNForDepthEstimation.from_pretrained("vinvino02/glpn-nyu")

    if image.mode !="RGB":
        image=image.convert("RGB")

    # load and resize the input image
    new_height = 480 if image.height > 480 else image.height
    new_height -= (new_height % 32)
    new_width = int(new_height * image.width / image.height)
    diff = new_width % 32
    new_width = new_width - diff if diff < 16 else new_width + 32 - diff
    new_size = (new_width, new_height)
    image = image.resize(new_size)

    # prepare image for the model
    inputs = feature_extractor(images=image, return_tensors="pt")

    # get the prediction from the model
    with torch.no_grad():
        outputs = model(**inputs)
        predicted_depth = outputs.predicted_depth

    output = predicted_depth.squeeze().cpu().numpy() * 1000.0

    # remove borders
    pad = 16
    output = output[pad:-pad, pad:-pad]
    image = image.crop((pad, pad, image.width - pad, image.height - pad))

    return image, output


def generate_mesh(image, depth_map, quality):
    width, height = image.size

    depth_image = (depth_map * 255 / np.max(depth_map)).astype('uint8')
    image = np.array(image)
    #print(depth_image)

    # create rgbd image
    depth_o3d = o3d.geometry.Image(depth_image)
    image_o3d = o3d.geometry.Image(image)
    rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(image_o3d, depth_o3d,
                                                                    convert_rgb_to_intensity=False)

    # camera settings
    camera_intrinsic = o3d.camera.PinholeCameraIntrinsic()
    camera_intrinsic.set_intrinsics(width, height, 500, 500, width / 2, height / 2)

    # create point cloud
    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image, camera_intrinsic)
    #o3d.visualization.draw_geometries([pcd])
    o3d.io.write_point_cloud("static/pcd_files/X_RAY.pcd", pcd)
    # o3d.io.write_point_cloud("X_RAY.xyz",pcd)

    # outliers removal
    cl, ind = pcd.remove_statistical_outlier(nb_neighbors=40, std_ratio=40.0)
    pcd = pcd.select_by_index(ind)

    # estimate normals
    pcd.estimate_normals()
    pcd.orient_normals_to_align_with_direction(orientation_reference=(0., 0., -1.))

    # surface reconstruction
    mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=15, n_threads=5)[0]
    '''
    points = np.genfromtxt('X_RAY.csv', delimiter=",", dtype=np.float32)
    point_cloud = pv.PolyData(points)
    mesh = point_cloud.reconstruct_surface()
    mesh.save('mesh.stl')    
'''
    # rotate the mesh
    rotation = mesh.get_rotation_matrix_from_xyz((np.pi, np.pi, 0))
    mesh.rotate(rotation, center=(0, 0, 0))

    # save the mesh
    temp_name = 'mesh'+ '.obj'
    o3d.io.write_triangle_mesh(f'static/obj_files/{temp_name}', mesh)
    #meshlab_cmd = 'meshlabserver -i mesh.obj -o mesh.obj'
    #subprocess.call(meshlab_cmd, shell=True)

    return temp_name
def model1(image):
    test_datagen = ImageDataGenerator(rescale=1. / 255)
    img_width, img_height = 224, 224
    img = image
    img = img.resize((img_width, img_height))
    x = np.array(img)
    x = x / 255.0
    x = np.expand_dims(x, axis=0)
    x = x.reshape(x.shape[0], x.shape[1], x.shape[2], 1)
    x = x.astype(np.float32)
    test_generator = test_datagen.flow(x, batch_size=1)
    #now we have to load the model
    new_model = tf.keras.models.load_model('knee_60.20.h5')

    y_pred = new_model.predict(test_generator)
    y_pred = list(y_pred)
    return y_pred.index(max(y_pred))

def model2_val(image):
    # First convert the image to cv2.imread file
    np_img = np.array(image)
    bgr_img = cv2.cvtColor(np_img, cv2.COLOR_RGB2BGR)
    _, buffer = cv2.imencode('.jpg', bgr_img)
    img = cv2.imdecode(np.frombuffer(buffer, np.uint8), cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # define a variable to carry all these values
    data_base = []

    # distance calculation
    img = cv2.equalizeHist(gray)
    img = cv2.GaussianBlur(img, (5, 5), 0)
    dist = 0
    thresh = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:2]
    centroids = []
    for c in contours:
        moments = cv2.moments(c)
        cx = int(moments['m10'] / moments['m00'])
        cy = int(moments['m01'] / moments['m00'])
        centroids.append((cx, cy))
    if len(centroids) < 2:
        dist = 0
    else:
        dist = np.sqrt((centroids[0][0] - centroids[1][0]) ** 2 + (centroids[0][1] - centroids[1][1]) ** 2)

    # Calculate the no of Contours
    blur = cv2.bilateralFilter(gray, 9, 75, 75)
    ret, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    thresh_inv = cv2.bitwise_not(thresh)

    kernel = np.ones((5, 5), np.uint8)
    closing = cv2.morphologyEx(thresh_inv, cv2.MORPH_CLOSE, kernel)
    opening = cv2.morphologyEx(closing, cv2.MORPH_OPEN, kernel)
    contours, hierarchy = cv2.findContours(opening, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    res = 0
    if contours:
        res = 1
    else:
        res = 0
    # Calculate no.0f oesteophytes and the area
    img = cv2.medianBlur(gray, 5)
    ret, thresh = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)
    contours, hierarchy = cv2.findContours(closing, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    areas = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        areas.append(area)
    no_oes = len(contours)
    area = np.mean(areas)

    # Calculate the bone density
    blur = cv2.bilateralFilter(gray, 9, 75, 75)
    ret, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    thresh_inv = cv2.bitwise_not(thresh)
    kernel = np.ones((5, 5), np.uint8)
    closing = cv2.morphologyEx(thresh_inv, cv2.MORPH_CLOSE, kernel)
    opening = cv2.morphologyEx(closing, cv2.MORPH_OPEN, kernel)
    bone_density = cv2.mean(blur, opening)[0]

    # Calculate the deformity in the bone
    img = cv2.medianBlur(gray, 5)
    thresh = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    min_contour_size = 1000
    max_contour_aspect_ratio = 0.5
    filtered_contours = []
    for contour in contours:
        contour_size = cv2.contourArea(contour)
        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = w / h
        if contour_size > min_contour_size and aspect_ratio < max_contour_aspect_ratio:
            filtered_contours.append(contour)
    joint_space_widths = []
    for contour in filtered_contours:
        _, _, w, _ = cv2.boundingRect(contour)
        joint_space_widths.append(w)
    average_joint_space_width = np.mean(joint_space_widths)
    jsw_value = 2.5
    deformity_of_bone = (average_joint_space_width - jsw_value) / jsw_value
    if str(deformity_of_bone) == 'nan':
        deformity_of_bone = 0
    else:
        deformity_of_bone = round(deformity_of_bone, 3)

    # Calculate the Angle of deviations in the bone
    gray = cv2.medianBlur(gray, 5)
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    kernel = np.ones((5, 5), np.uint8)
    dilated = cv2.dilate(edges, kernel, iterations=1)
    eroded = cv2.erode(dilated, kernel, iterations=1)
    contours, hierarchy = cv2.findContours(eroded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) < 2:
        angle_degrees = 0
    else:
        sorted_contours = sorted(contours, key=cv2.contourArea, reverse=True)
        cnt1 = sorted_contours[0]
        cnt2 = sorted_contours[1]
        rect1 = cv2.minAreaRect(cnt1)
        box1 = cv2.boxPoints(rect1)
        box1 = np.int0(box1)
        rect2 = cv2.minAreaRect(cnt2)
        box2 = cv2.boxPoints(rect2)
        box2 = np.int0(box2)
        angle1 = np.arctan2(box1[0][1] - box1[1][1], box1[0][0] - box1[1][0])
        angle2 = np.arctan2(box2[0][1] - box2[1][1], box2[0][0] - box2[1][0])
        angle_degrees = np.abs(np.degrees(angle1 - angle2))
    data_base.append([dist, res, no_oes, area, bone_density, deformity_of_bone, angle_degrees])
    return data_base


def water_content(image):
    mesh = trimesh.load('static/obj_files/mesh.obj')
    # Extract the vertices and faces of the mesh
    vertices = mesh.vertices
    faces = mesh.faces
    # Calculate the average Z-coordinate of each face
    z_coords = vertices[:, 2][faces]
    z_avg = np.mean(z_coords, axis=1)
    # Sort the faces by average Z-coordinate
    sorted_faces = faces[np.argsort(z_avg)]
    # Extract the top N faces with the highest average Z-coordinate
    # These faces correspond to the bone tissue
    N = 20
    bone_faces = sorted_faces[-N:]
    # Create a binary mask of the bone tissue region
    mask = np.zeros_like(mesh.visual.vertex_colors)
    gray_mask = 0
    # Check the number of channels in the input image
    if len(mask.shape) == 2:
        # Grayscale image with only one channel
        # Skip the color conversion step
        gray_mask = mask
    else:
        # Color image with more than one channel
        # Convert to grayscale
        gray_mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

    # Load the X-ray image
    # First convert the image to cv2.imread file
    np_img = np.array(image)
    bgr_img = cv2.cvtColor(np_img, cv2.COLOR_RGB2BGR)
    _, buffer = cv2.imencode('.jpg', bgr_img)
    img = cv2.imdecode(np.frombuffer(buffer, np.uint8), cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    gray_mask = cv2.resize(gray_mask, (gray.shape[1], gray.shape[0]), interpolation=cv2.INTER_NEAREST)
    # Segment the bone and surrounding soft tissue
    # Threshold the image to separate the bone tissue from the soft tissue
    thresh_val, binary_img = cv2.threshold(img, 100, 255, cv2.THRESH_BINARY)

    # Apply the bone tissue mask to the binary image
    masked_img = cv2.bitwise_and(binary_img, binary_img, mask=gray_mask)

    # Calculate the water content
    # Extract the soft tissue region from the masked image
    soft_tissue = np.multiply(img, 1 - masked_img / 255.0)

    # Compute the mean intensity of the soft tissue region
    mean_intensity = np.mean(soft_tissue)

    # Calculate the water content as a percentage of the maximum possible water content
    # This assumes that the soft tissue region contains only water and no other materials
    water_content = (mean_intensity / 255.0) * 100.0

    return water_content

