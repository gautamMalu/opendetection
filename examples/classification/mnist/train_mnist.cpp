#include <detectors/global2D/training/ODConvTrainer.h>
#include<string>

using namespace od;
using namespace std;

int main(int argc, char **argv)
{
	od::g2d::ODConvTrainer *mnist_trainer = new od::g2d::ODConvTrainer("","");
	//TODO: Add usage example doumentation here	
	if (argc < 2){ 
		mnist_trainer->setSolverParameters("examples/classification/mnist/lenet_train_test.prototxt",
                                                0.005, "fixed",10000,
                                                5000, "examples/classification/mnist/lenet");
		std::cout << "Staring Training with following parameters" << std::endl;
		mnist_trainer->getSolverParameters();
		mnist_trainer->startTraining();
	}
	else if (argc == 2){
		mnist_trainer->setSolverParametersFromFile(argv[1]);
		std::cout << "Training started with paramters from " << argv[1] <<std::endl;
		std::cout << "Staring Training with following parameters" << std::endl;
                mnist_trainer->getSolverParameters();
		mnist_trainer->startTraining();
	}
	else if (argc == 3){
		mnist_trainer->setSolverParametersFromFile(argv[1]);
		std::cout << "Training started with paramters from " << argv[1] <<std::endl;
		std::cout << "Staring Training with following parameters" << std::endl;
                mnist_trainer->getSolverParameters();
		std::string arg2(argv[2]);
		if(arg2.compare(arg2.size() - 11, 11, ".caffemodel") == 0){
			std::cout << "Finetuning with "<< arg2 << std::endl;
			mnist_trainer->fineTuning(arg2);
		}
		else{
			std::cout << "Resuming training with " << arg2 << std::endl;
			mnist_trainer->resumeTraining(arg2);
		}
	}
	return 0;
}
	
