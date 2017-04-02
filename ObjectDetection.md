Object Recognition
=============
![Object Recognition](http://host.robots.ox.ac.uk/pascal/VOC/voc2007/examples/bottle_05.jpg)

Usually there are fixed number of classes which an algorithm can recognize (Car, dog, Human, Bike etc). It depends on which dataset the algorithm has been trained. There are two dataset on which are used for benchmarking of object recognition algorithms. Early algorithms have focused only on single object class such as face detection, or pedestrian detection. Here is the summary table of benchmarking datasets:n

| Dataset Name        | # object categories           | # Training Images  |Website |
| ------------- |:-------------:| -----:| ---------------:|
| PASCAL VOC 2012     | 20 |  11,530  | http://host.robots.ox.ac.uk/pascal/VOC/ |
| MS COCO     | 80      |   300,000  |    http://mscoco.org/home/   |

Most of the object detection techniques can be summerized as follows:

* Deformable Parts Model: Sliding Window Approach. Examples: HOG based face detectors in opencv. 
* Region Proposal Method: First generate the potential bouding boxes, then run classifier over these windows. Examples: Fast-RCNNs, Faster-RCNN, Multi-BOx etc. Implementation of faster RCNN is availabe for caffe.
* Single Shot Detection: Simultaeously predicts bounding boxes and class probability scores for these predicted boxes in a single forward pass. Examples: SSD, YOLO. Implementation of SSD, and YOLO are availabe for caffe.





