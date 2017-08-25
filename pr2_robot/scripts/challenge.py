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
from sensor_msgs.msg import JointState
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

def pcl_filtering(pcl_msg):
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

    # Convert PCL data to ROS messages
    ros_pcl_objects = pcl_to_ros(cloud_objects)
    ros_pcl_table = pcl_to_ros(cloud_table)
    ros_cluster_cloud = pcl_to_ros(cluster_cloud)

    # Publish ROS messages
    pcl_objects_pub.publish(ros_pcl_objects)
    pcl_table_pub.publish(ros_pcl_table)
    pcl_cluster_pub.publish(ros_cluster_cloud)

    # Create collision point
    collision_point = {}
    collision_point["table"] = cloud_table.to_array()
    collision_point_pub.publish(ros_pcl_table)

    # Exercise-3 TODOs:
    # Classify the clusters! (loop through each detected cluster one at a time)
    detected_objects_labels_all = []
    detected_objects_all = []
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
        # Add the detected object to the collision map
        # collision_point[label] = pcl_cluster.to_array()

        if not any(item.label == label for item in detected_objects_all):
        # for x in detected_objects_all:
        #     if label in x.label:
            detected_objects_all.append(do)
            detected_objects_labels_all.append(label)

    rospy.loginfo('Detected {} objects: {}'.format(len(detected_objects_labels), detected_objects_labels))
    rospy.loginfo('All detected {} objects: {}'.format(len(detected_objects_labels_all), detected_objects_labels_all))

    # Publish the list of detected objects
    detected_objects_pub.publish(detected_objects)
    detected_objects_all_pub.publish(detected_objects_all)

    return detected_objects, collision_point

def pr2_mov(rad):
    '''move pr2 world joint to desired angle (rad)'''
    rate = rospy.Rate(50) # 50hz
    world_joint_pub.publish(rad)
    rate.sleep()

    joint_state = rospy.wait_for_message('/pr2/joint_states', JointState)

    return joint_state.position[19]

def pr2_rot():
    ''' rotate pr2 right and left to detect environment'''
    global rotation_state
    global rot_dir

    if rotation_state:
        if rot_dir == 'left':
            world_joint_state = pr2_mov(1.57)
            if np.abs(world_joint_state - 1.57) < 0.1:                
                rot_dir = 'right'
                print("Get left side, go to right side now...")

        if rot_dir == 'right':
            world_joint_state = pr2_mov(-1.57)
            if np.abs(world_joint_state + 1.57) < 0.1:                
                rot_dir = 'center'
                print("Get right side, go to center now...")

        if rot_dir == 'center':
            world_joint_state = pr2_mov(0)
            if np.abs(world_joint_state) < 0.1:                
                rotation_state = False
                print("Get center, exist rotation.")
            
def pcl_callback(pcl_msg):
    '''Callback function for your Point Cloud Subscriber'''
    
    # Rotate PR2 in place to capture side tables for the collision map
    collision_map = False
    if collision_map:
        pr2_rot()
    else:
        world_joint_state = pr2_mov(0)

    detected_objects, collision_point = pcl_filtering(pcl_msg)
    
    # Suggested location for where to invoke your pr2_mover() function within pcl_callback()
    # Could add some logic to determine whether or not your object detections are robust
    # before calling pr2_mover()
    try:
        if len(detected_objects) > 0:
            pr2_mover(detected_objects, collision_point)
    except rospy.ROSInterruptException:
        pass


def pr2_mover(object_list, collision_point=None):
    '''function to load parameters and request PickPlace service'''
    # Initialize variables
    global use_collision_map

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

    # Loop through the pick list
    target_count_left = 0
    target_count_right = 0
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
                dropbox_x = -0.1 #dropbox_position[0]
                # Add olace pose bias for each object
                if arm_name.data == 'right':
                    dropbox_y = dropbox_position[1] - 0.10 + target_count_right*0.1                  
                else:
                    dropbox_y = dropbox_position[1] - 0.10 + target_count_left*0.03
                dropbox_z = dropbox_position[2] + 0.1
                place_pose.position.x = np.float(dropbox_x)
                place_pose.position.y = np.float(dropbox_y)
                place_pose.position.z = np.float(dropbox_z)            

        # Create a list of dictionaries (made with make_yaml_dict()) for later output to yaml format
        yaml_dict = make_yaml_dict(test_scene_num, arm_name, object_name, pick_pose, place_pose)
        dict_list.append(yaml_dict)

        # if use_collision_map:
        #     # Delete the target clound from collision map
        #     del collision_point[target.label]

        #     # Creating collision map
        #     points_list = np.empty((0,4), float)
        #     for index, target_pts in collision_point.iteritems():
        #         points_list = np.append(points_list, target_pts[:,:4], axis=0)

        #     collision_cloud = pcl.PointCloud_PointXYZRGB()
        #     collision_cloud.from_list(np.ndarray.tolist(points_list))
        #     collision_point_pub.publish(pcl_to_ros(collision_cloud))
        #     rospy.sleep(2)

        # Wait for 'pick_place_routine' service to come up
        print("Target now: ", target.label)
        rospy.wait_for_service('pick_place_routine')
        try:
            pick_place_routine = rospy.ServiceProxy('pick_place_routine', PickPlace)
            # Insert message variables to be sent as a service request   
            resp = pick_place_routine(test_scene_num, object_name, arm_name, pick_pose, place_pose)
            print ("Response: ",resp.success)
            # Count number to set bias value for the object arrangement
            if resp.success:
                if arm_name.data == 'right':
                    target_count_right += 1
                    if target_count_right == 3:
                        target_count_right = 0.5
                else:
                    target_count_left += 1

        except rospy.ServiceException, e:
            print "Service call failed: %s"%e

    # Output request parameters into output yaml file  
    yaml_filename = 'output_' + str(test_scene_num.data) + '.yaml'
    if not os.path.exists(yaml_filename):
        send_to_yaml(yaml_filename, dict_list)
        print(yaml_filename + "has been saved.")
    # else:
    #     print(yaml_filename + "has been existed.")


if __name__ == '__main__':
    # Initial global variables
    rotation_state = True
    world_joint_state = 0
    rot_dir = 'left'
    use_collision_map = True

    # ROS node initialization
    rospy.init_node('perception_project', anonymous=True)

    # Create Subscribers
    pcl_sub = rospy.Subscriber("/pr2/world/points", pc2.PointCloud2, pcl_callback, queue_size=1)
    # joint_sub = rospy.Subscriber("/joint_state", JointState, state_callback, queue_size=1)

    # Create Publishers
    pcl_objects_pub = rospy.Publisher("/pcl_objects", PointCloud2, queue_size=1)
    pcl_table_pub = rospy.Publisher("/pcl_table", PointCloud2, queue_size=1)
    pcl_cluster_pub = rospy.Publisher("/pcl_cluster", PointCloud2, queue_size=1)
    
    object_markers_pub = rospy.Publisher("/object_markers", Marker, queue_size=1)
    detected_objects_pub = rospy.Publisher("/detected_objects", DetectedObjectsArray, queue_size=1)
    detected_objects_all_pub = rospy.Publisher("/detected_objects_all", DetectedObjectsArray, queue_size=1)

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
