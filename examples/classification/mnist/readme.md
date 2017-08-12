This example showcase how to use the ODConvTrainer module. 
In this example we create an mnist trainer object and use that to 
train lenet on mnist data. The data is prepared from caffe's scripts. 
The solver and model definitation files are also borrowed from caffe's mnist example.

Assuming your build location is `opendetection\build` and you are running this example
from source directory i.e. `opendetection`  

run this command to download mnist  
`./examples/classification/mnist/get_mnist.sh`  
NOTE: You would need wget and gunzip to download mnist data and to unzip it.  

run this command to covert mnist data into LMDB format which will be used 
for training in caffe.  
`./examples/classification/mnist/create_mnist.sh`  

run this command to start lenet traning of mnist:    
`./build/examples/classification/mnist/train_mnist examples/mnist/lenet_solver.prototxt`    

to resume training from solverstate file:      
`./build/examples/classification/mnist/train_mnist examples/mnist/lenet_solver.prototxt examples/mnist/lenet_iter_5000.solverstate`  

to finetune from pretrained weights:      
`./build/examples/classification/mnist/train_mnist examples/mnist/lenet_solver.prototxt examples/mnist/lenet_iter_5000.caffemodel` 
