#include <detectors/global2D/training/ODConvTrainer.h>
#include<string>

using namespace od;
using namespace std;

int main(int argc, char **argv)
{
	od::g2d::ODConvTrainer *cub_trainer = new od::g2d::ODConvTrainer("","");
	/* usage: ./build/examples/classification/cub/train_cub ./examples/classifcation/cub/solver.prototxt
	   ./examples/classifcation/cub/bvlc_reference_caffenet.caffemodel	*/	
	if (argc > 2){ 
		cub_trainer->setSolverParametersFromFile(argv[1]);
		std::cout << "Solver paramters from " << argv[1] << " are:" << std::endl;
		cub_trainer->getSolverParameters();
		std::string arg2(argv[2]);
		std::cout << "finetuning with " << arg2 << std::endl;
		cub_trainer->fineTuning(arg2);
	}
	else{
		std::cout << "usage: ./build/examples/classification/cub/train_cub \
			./examples/classifcation/cub/solver.prototxt \
			./examples/classifcation/cub/bvlc_reference_caffenet.caffemodel" \
			<< std::endl;
	}
	return 0;
}	
