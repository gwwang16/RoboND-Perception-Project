#!/usr/bin/env python

# Import modules
import numpy as np
import sklearn
from sklearn.preprocessing import LabelEncoder
import pickle
from sensor_stick.srv import GetNormals
from sensor_stick.features import compute_color_histograms
from sensor_stick.features import compute_normal_histograms
from visualization_msgs.msg import Marker
from sensor_stick.marker_tools import *
from sensor_stick.msg import DetectedObjectsArray
from sensor_stick.msg import DetectedObject
from sensor_stick.pcl_helper import *

import rospy
import tf
from geometry_msgs.msg import Pose
from std_msgs.msg import Float64
from std_msgs.msg import Int32
from std_msgs.msg import String
from pr2_robot.srv import *
from rospy_message_converter import message_converter
import yaml
import os

def get_normals(cloud):
    '''Helper function to get surface normals'''
    get_normals_prox = rospy.ServiceProxy('/feature_extractor/get_normals', GetNormals)
    return get_normals_prox(cloud).cluster

def make_yaml_dict(test_scene_num, arm_name, object_name, pick_pose, place_pose):
    '''Helper function to create a yaml friendly dictionary from ROS messages'''
    yaml_dict = {}
    yaml_dict["test_scene_num"] = test_scene_num.data
    yaml_dict["arm_name"]  = arm_name.data
    yaml_dict["object_name"] = object_name.data
    yaml_dict["pick_pose"] = message_converter.convert_ros_message_to_dictionary(pick_pose)
    yaml_dict["place_pose"] = message_converter.convert_ros_message_to_dictionary(place_pose)
    return yaml_dict

def send_to_yaml(yaml_filename, dict_list):
    '''Helper function to output to yaml file'''
    data_dict = {"object_list": dict_list}
    with open(yaml_filename, 'w') as outfile:
        yaml.dump(data_dict, outfile, default_flow_style=False)

def pcl_callback(pcl_msg):
    '''Callback function for your Point Cloud Subscriber'''
    ###### Exercise-2 TODOs: #####
    # Convert ROS msg to PCL data
    cloud = ros_to_pcl(pcl_msg)

    # Statistical Outlier Filtering
    # Create statistical outlier filter object
    outlier_filter = cloud.make_statistical_outlier_filter()
    # Set the number of neighboring points
    outlier_filter.set_mean_k(5)
    # Set threshold scale factor
    x = 0.1
    # Mean distance larger than (mean distance+x*std_dev) will be considered as outlier
    outlier_filter.set_std_dev_mul_thresh(x)
    cloud_filtered = outlier_filter.filter()

    # Voxel Grid Downsampling
    # Create a voxelgrid filter object for our input point cloud
    vox = cloud_filtered.make_voxel_grid_filter()
    # Choose a voxel size (leaf size)
    LEAF_SIZE = 0.005
    # Set the voxel size
    vox.set_leaf_size(LEAF_SIZE, LEAF_SIZE, LEAF_SIZE)
    cloud_filtered = vox.filter()

    # PassThrough Filter in z axis
    passthrough = cloud_filtered.make_passthrough_filter()
    # Assign axis and range to the passthrough filter object
    filter_axis = 'z'
    passthrough.set_filter_field_name(filter_axis)
    axis_min = 0.6
    axis_max = 1.0
    passthrough.set_filter_limits(axis_min, axis_max)
    # Finally use the filter function to obtain the resultant point cloud.
    cloud_filtered = passthrough.filter()

    # PassThrough Filter in y axis
    passthrough_y = cloud_filtered.make_passthrough_filter()
    # Assign axis and range to the passthrough filter object
    filter_axis = 'y'
    passthrough_y.set_filter_field_name(filter_axis)
    axis_min = -0.5
    axis_max = 0.5
    passthrough_y.set_filter_limits(axis_min, axis_max)
    # Finally use the filter function to obtain the resultant point cloud.
    cloud_filtered = passthrough_y.filter()

    # RANSAC Plane Segmentation
    # Create the segmentation object
    seg = cloud_filtered.make_segmenter()
    # Set the model you wish to fit
    seg.set_model_type(pcl.SACMODEL_PLANE)
    seg.set_method_type(pcl.SAC_RANSAC)
    # Max distance for a point to be considered fitting the model
    max_distance = 0.01
    seg.set_distance_threshold(max_distance)
    # Call the segment function to obtain set of inlier indices and model coefficients
    inliers, coefficients = seg.segment()

    # Extract inliers and outliers
    cloud_table = cloud_filtered.extract(inliers, negative=False)
    cloud_objects = cloud_filtered.extract(inliers, negative=True)

    # Euclidean Clustering
    white_cloud = XYZRGB_to_XYZ(cloud_objects)
    tree = white_cloud.make_kdtree()
    # Create a cluster extraction object
    ec = white_cloud.make_EuclideanClusterExtraction()
    # Set tolerances for distance threshold
    # as well as minimum and maximum cluster size (in points)
    ec.set_ClusterTolerance(0.01)
    ec.set_MinClusterSize(100)
    ec.set_MaxClusterSize(3000)
    # Search the k-d tree for clusters
    ec.set_SearchMethod(tree)
    # Extract indices for each of the discovered clusters
    cluster_indices = ec.Extract()

    # Create Cluster-Mask Point Cloud to visualize each cluster separately
    # Assign a color corresponding to each segmented object in scene
    cluster_color = get_color_list(len(cluster_indices))
    color_cluster_point_list = []
    for j, indices in enumerate(cluster_indices):
        for i, indice in enumerate(indices):
            color_cluster_point_list.append([white_cloud[indice][0],
                                             white_cloud[indice][1],
                                             white_cloud[indice][2],
                                             rgb_to_float(cluster_color[j])])
    # Create new cloud containing all clusters, each with unique color
    cluster_cloud = pcl.PointCloud_PointXYZRGB()
    cluster_cloud.from_list(color_cluster_point_list)

    # Create collision point
    collision_point = cloud_table
    # TODO: add reminding objects into collision map!!

    # Convert PCL data to ROS messages
    ros_pcl_objects = pcl_to_ros(cloud_objects)
    ros_pcl_table = pcl_to_ros(cloud_table)
    ros_cluster_cloud = pcl_to_ros(cluster_cloud)
    ros_collision_point = pcl_to_ros(collision_point)

    # Publish ROS messages
    pcl_objects_pub.publish(ros_pcl_objects)
    pcl_table_pub.publish(ros_pcl_table)
    pcl_cluster_pub.publish(ros_cluster_cloud)
    collision_point_pub.publish(ros_collision_point)

    # Exercise-3 TODOs:

    # Classify the clusters! (loop through each detected cluster one at a time)
    detected_objects_labels = []
    detected_objects = []
    for index, pts_list in enumerate(cluster_indices):
        # Grab the points for the cluster
        pcl_cluster = cloud_objects.extract(pts_list)

        ros_cluster = pcl_to_ros(pcl_cluster)

        # Compute the associated feature vector
        chists = compute_color_histograms(ros_cluster, using_hsv=True)
        normals = get_normals(ros_cluster)
        nhists = compute_normal_histograms(normals)
        feature = np.concatenate((chists, nhists[:1]))

        # Check shapes and values of normals 
        # norm_x_vals = []
        # norm_y_vals = []
        # norm_z_vals = []
        # for norm_component in pc2.read_points(normals, field_names = ('normal_x', 'normal_y', 'normal_z'),
        #                                   skip_nans=True):
        #     norm_x_vals.append(norm_component[0])
        #     norm_y_vals.append(norm_component[1])
        #     norm_z_vals.append(norm_component[2])
        # print("mean:")
        # print(np.mean(norm_x_vals), np.mean(norm_y_vals), np.mean(norm_z_vals))
        # print("shape:")
        # print(np.shape(norm_x_vals), np.shape(norm_y_vals), np.shape(norm_z_vals))
        # print(np.min(norm_x_vals), np.min(norm_y_vals), np.min(norm_z_vals))
        # print(np.max(norm_x_vals), np.max(norm_y_vals), np.max(norm_z_vals))

        # Make the prediction
        prediction = clf.predict(scaler.transform(feature.reshape(1,-1)))
        label = encoder.inverse_transform(prediction)[0]
        detected_objects_labels.append(label)

        # Publish a label into RViz
        label_pos = list(white_cloud[pts_list[0]])
        label_pos[2] += .4
        object_markers_pub.publish(make_label(label,label_pos, index))

        # Add the detected object to the list of detected objects.
        do = DetectedObject()
        do.label = label
        do.cloud = ros_cluster
        detected_objects.append(do)

    rospy.loginfo('Detected {} objects: {}'.format(len(detected_objects_labels), detected_objects_labels))


    # Publish the list of detected objects
    detected_objects_pub.publish(detected_objects)

    # Suggested location for where to invoke your pr2_mover() function within pcl_callback()
    # Could add some logic to determine whether or not your object detections are robust
    # before calling pr2_mover()
    try:
        pr2_mover(detected_objects)
    except rospy.ROSInterruptException:
        pass


def pr2_mover(object_list):
    '''function to load parameters and request PickPlace service'''
    # Initialize variables
    dict_list = []
    labels = []
    centroids = []
    object_list_param = []
    dropbox_param = []
    pick_position = []
    dropbox_position = []

    test_scene_num = Int32()
    object_name = String()
    arm_name = String()
    pick_pose = Pose()
    place_pose = Pose()

    test_scene_num.data = 3    

    # Read yaml parameters
    object_list_param = rospy.get_param('/object_list')
    dropbox_param = rospy.get_param('/dropbox')
 
    # TODO: Rotate PR2 in place to capture side tables for the collision map
    # world_joint_pub.publish(1.57)
    # rospy.sleep(10)
    # world_joint_pub.publish(-1.57)
    # rospy.sleep(25)
    # world_joint_pub.publish(0)
    # rospy.sleep(5)

    # Loop through the pick list
    for target in object_list:
        
        labels.append(target.label)
        points_arr = ros_to_pcl(target.cloud).to_array()
        pick_position = np.mean(points_arr, axis=0)[:3]
        pick_pose.position.x = np.float(pick_position[0])
        pick_pose.position.y = np.float(pick_position[1])
        pick_pose.position.z = np.float(pick_position[2])
        centroids.append(pick_position[:3])

        object_name.data = str(target.label)

        # Assign the arm and 'place_pose' to be used for pick_place
        for index in range(0, len(object_list_param)):
            if object_list_param[index]['name'] == target.label:
                object_group = object_list_param[index]['group']
        for ii in range(0, len(dropbox_param)):
            if dropbox_param[ii]['group'] == object_group:
                arm_name.data = dropbox_param[ii]['name']
                dropbox_position = dropbox_param[ii]['position']
                dropbox_x = dropbox_position[0]
                dropbox_y = dropbox_position[1]
                dropbox_z = dropbox_position[2]
                place_pose.position.x = np.float(dropbox_x)
                place_pose.position.y = np.float(dropbox_y)
                place_pose.position.z = np.float(dropbox_z)            

        # Create a list of dictionaries (made with make_yaml_dict()) for later output to yaml format
        yaml_dict = make_yaml_dict(test_scene_num, arm_name, object_name, pick_pose, place_pose)
        dict_list.append(yaml_dict)

        # Wait for 'pick_place_routine' service to come up
        # rospy.wait_for_service('pick_place_routine')

        # try:
        #     pick_place_routine = rospy.ServiceProxy('pick_place_routine', PickPlace)

        #     # Insert message variables to be sent as a service request   
        #     resp = pick_place_routine(test_scene_num, object_name, arm_name, pick_pose, place_pose)
        #     print ("Response: ",resp.success)

        # except rospy.ServiceException, e:
        #     print "Service call failed: %s"%e

    # Output request parameters into output yaml file  
    yaml_filename = 'output_' + str(test_scene_num.data) + '.yaml'
    if not os.path.exists(yaml_filename):
        send_to_yaml(yaml_filename, dict_list)
        print(yaml_filename + "has been saved.")
    else:
        print(yaml_filename + "has been existed.")


if __name__ == '__main__':

    # ROS node initialization
    rospy.init_node('perception_project', anonymous=True)

    # Create Subscribers
    pcl_sub = rospy.Subscriber("/pr2/world/points", pc2.PointCloud2, pcl_callback, queue_size=1)

    # Create Publishers
    pcl_objects_pub = rospy.Publisher("/pcl_objects", PointCloud2, queue_size=1)
    pcl_table_pub = rospy.Publisher("/pcl_table", PointCloud2, queue_size=1)
    pcl_cluster_pub = rospy.Publisher("/pcl_cluster", PointCloud2, queue_size=1)
    
    object_markers_pub = rospy.Publisher("/object_markers", Marker, queue_size=1)
    detected_objects_pub = rospy.Publisher("/detected_objects", DetectedObjectsArray, queue_size=1)

    collision_point_pub = rospy.Publisher("/pr2/3d_map/points", PointCloud2, queue_size=1)

    world_joint_pub = rospy.Publisher("/pr2/world_joint_controller/command", Float64, queue_size=10)

    # Load Model From disk
    model = pickle.load(open('model.sav', 'rb'))
    clf = model['classifier']
    encoder = LabelEncoder()
    encoder.classes_ = model['classes']
    scaler = model['scaler']

    # Initialize color_list
    get_color_list.color_list = []

    # Spin while node is not shutdown
    while not rospy.is_shutdown():
        rospy.spin()
