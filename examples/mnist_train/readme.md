This example showcase how to use the ODConvTrainer module. 
In this example we create an mnist trainer object and use that to 
train lenet on mnist data. The data is prepared from caffe's scripts. 
The solver and model definitation files are also borrowed from caffe's mnist example.

Assuming your build location is `opendetection\build` and you are running this example
from source directory i.e. `opendetection` run this command to start lenet traning of mnist:    
`./build/examples/mnist_train/mnist_train examples/mnist_train/lenet_solver.prototxt`    

to resume training from solverstate file:      
`./build/examples/mnist_train/mnist_train examples/mnist_train/lenet_solver.prototxt examples/mnist_train/lenet_iter_5000.solverstate`  

to finetune from pretrained weights:      
`./build/examples/mnist_train/mnist_train examples/mnist_train/lenet_solver.prototxt examples/mnist_train/lenet_iter_5000.caffemodel` 
