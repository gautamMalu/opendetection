#include <detectors/global2D/training/ODConvTrainer.h>
//#include "detectors/global2D/detection/ODConvClassification.h"
#include<string>
#include "common/pipeline/ObjectDetector.h"
#include "common/pipeline/ODDetection.h"


using namespace od;
using namespace std;

int main(int argc, char **argv)
{
	od::g2d::ODConvTrainer *mnist_trainer = new od::g2d::ODConvTrainer("","");
	//TODO: Add usage example doumentation here	
	if (argc < 2){ 
		std::cout << "please provide solver file" << std::endl;
	}
	else if (argc == 2){
		mnist_trainer->setSolverLocation(argv[1]);
		std::cout << "Training started with " << argv[1] <<std::endl;
		mnist_trainer->startTraining();
	}
	else if (argc == 3){
		mnist_trainer->setSolverLocation(argv[1]);
		std::cout << "Got solver file as " << argv[1] << std::endl;
		std::string arg(argv[2]);
		if(arg.compare(arg.size() - 11, 11, ".caffemodel") == 0){
			std::cout << "Finetuning with "<< arg << std::endl;
			mnist_trainer->fineTuning(arg);
		}
		else{
			std::cout << "Resuming trainig with " << arg << std::endl;
			mnist_trainer->resumeTraining(arg);
		}
	}
	return 0;
}
	
