## Project: Perception Pick & Place
---


# Required Steps for a Passing Submission:
1. Extract features and train an SVM model on new objects (see `pick_list_*.yaml` in `/pr2_robot/config/` for the list of models you'll be trying to identify).
2. Write a ROS node and subscribe to `/pr2/world/points` topic. This topic contains noisy point cloud data that you must work with.
3. Use filtering and RANSAC plane fitting to isolate the objects of interest from the rest of the scene.
4. Apply Euclidean clustering to create separate clusters for individual items.
5. Perform object recognition on these objects and assign them labels (markers in RViz).
6. Calculate the centroid (average in x, y and z) of the set of points belonging to that each object.
7. Create ROS messages containing the details of each object (name, pick_pose, etc.) and write these messages out to `.yaml` files, one for each of the 3 scenarios (`test1-3.world` in `/pr2_robot/worlds/`).  See the example `output.yaml` for details on what the output should look like.
8. Submit a link to your GitHub repo for the project or the Python code for your perception pipeline and your output `.yaml` files (3 `.yaml` files, one for each test world).  You must have correctly identified 100% of objects from `pick_list_1.yaml` for `test1.world`, 80% of items from `pick_list_2.yaml` for `test2.world` and 75% of items from `pick_list_3.yaml` in `test3.world`.
9. Congratulations!  Your Done!

# Extra Challenges: Complete the Pick & Place
7. To create a collision map, publish a point cloud to the `/pr2/3d_map/points` topic and make sure you change the `point_cloud_topic` to `/pr2/3d_map/points` in `sensors.yaml` in the `/pr2_robot/config/` directory. This topic is read by Moveit!, which uses this point cloud input to generate a collision map, allowing the robot to plan its trajectory.  Keep in mind that later when you go to pick up an object, you must first remove it from this point cloud so it is removed from the collision map!
8. Rotate the robot to generate collision map of table sides. This can be accomplished by publishing joint angle value(in radians) to `/pr2/world_joint_controller/command`
9. Rotate the robot back to its original state.
10. Create a ROS Client for the “pick_place_routine” rosservice.  In the required steps above, you already created the messages you need to use this service. Checkout the [PickPlace.srv](https://github.com/udacity/RoboND-Perception-Project/tree/master/pr2_robot/srv) file to find out what arguments you must pass to this service.
11. If everything was done correctly, when you pass the appropriate messages to the `pick_place_routine` service, the selected arm will perform pick and place operation and display trajectory in the RViz window
12. Place all the objects from your pick list in their respective dropoff box and you have completed the challenge!
13. Looking for a bigger challenge?  Load up the `challenge.world` scenario and see if you can get your perception pipeline working there!


[//]: # "Image References"
[confusion_matrix_1]: ./pictures/confusion_matrix_1.png
[confusion_matrix_2]: ./pictures/confusion_matrix_2.png
[pick_list_1]: ./pictures/pick_list_1.jpg
[pick_list_1_result]: ./pictures/pick_list_1_result.jpg
[pick_list_2]: ./pictures/pick_list_2.jpg
[pick_list_2_result]: ./pictures/pick_list_2_result.jpg
[pick_list_3]: ./pictures/pick_list_3.jpg
[pick_list_3_result]: ./pictures/pick_list_3_result.jpg


## [Rubric](https://review.udacity.com/#!/rubrics/1067/view) Points

### Exercise 1, 2 and 3 pipeline implemented
#### 1. Complete Exercise 1 steps. Pipeline for filtering and RANSAC plane fitting implemented.
- Statistical Outlier Filtering
    Set the number of neighboring points 5 and threshold scale factor 0.1, any points with mean distance larger than (mean distance+x\*std_dev ) will be considered as outlier.
```
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
```
- Downsampling voxel grid
    `LEAF_SIZE` is set as 0.005
```    
    # Voxel Grid Downsampling
    # Create a voxelgrid filter object for our input point cloud
    vox = cloud_filtered.make_voxel_grid_filter()
    # Choose a voxel size (leaf size)
    LEAF_SIZE = 0.005
    # Set the voxel size
    vox.set_leaf_size(LEAF_SIZE, LEAF_SIZE, LEAF_SIZE)
    cloud_filtered = vox.filter()
```
- PassThrough Filter
    Create Passthrough filter in y and z axes
```    
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
```
- RANSAC Plane Segmentation
```
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
```

#### 2. Complete Exercise 2 steps: Pipeline including clustering for segmentation implemented.

```
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
```

#### 3. Complete Exercise 3 Steps.  Features extracted and SVM trained.  Object recognition implemented.

- Features extracted

```
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
            detected_objects_all.append(do)
            detected_objects_labels_all.append(label)
```

- SVM trained

  - In `features.py` (sensor_stick/src/sensor_stick):

    64 bins with range (0, 256) to compute color histograms, 

    3 bins with range (-1, 1) to compute normal histograms.

  - in `pick_list_3.yaml` (src/RoboND-Perception-Project/pr2_robot/config):

    50 features were captured for each object  to train SVM classifier. `LinearSVC` classifier is adopted here,  in which `l2` regularization method is used to avoid over fitting problem

```
	clf = svm.LinearSVC(penalty='l2', loss='squared_hinge', dual=False, tol=1e-4)
```

​		`r2` accuracy scoring is used in `cross_val_score()` to increase accuracy.

```
	scores = cross_validation.cross_val_score(cv=kf, estimator=clf,
                                         	X=X_train, y=y_train,
                                         	scoring='r2')
```

​	The confusion matrix without normalization is

![alt text][confusion_matrix_1]

​	The normalized confusion matrix is

![alt text][confusion_matrix_2]

-  Object recognition

   Variables Initialization
```
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
```

​     Read objects and dropbox params from yaml files
```
    object_list_param = rospy.get_param('/object_list')
    dropbox_param = rospy.get_param('/dropbox')   
```
​     Loop through each object in the pick list, and ten assign the arm and 'place_pose' to be used for pick_place, create a list of dictionaries for later output to yaml file.
```
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
        # Create a list of dictionaries for later output to yaml format
        yaml_dict = make_yaml_dict(test_scene_num, arm_name, object_name, pick_pose, place_pose)
        dict_list.append(yaml_dict)
```



### Pick and Place Setup

#### 1. For all three tabletop setups (`test*.world`), perform object recognition, then read in respective pick list (`pick_list_*.yaml`). Next construct the messages that would comprise a valid `PickPlace` request output them to `.yaml` format.
- Output request parameters into output yaml file  
```
    yaml_filename = 'output_' + str(test_scene_num.data) + '.yaml'
    if not os.path.exists(yaml_filename):
        send_to_yaml(yaml_filename, dict_list)
```

- Object recognition results

![alt text][pick_list_1]
![alt text][pick_list_2]
![alt text][pick_list_3]

### Extra Challenges: 
![alt text][pick_list_1_result]

![alt text][pick_list_2_result]

![alt text][pick_list_3_result]