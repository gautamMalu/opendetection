#include <detectors/global2D/detection/ODConvClassifier.h>
#include <common/pipeline/ODScene.h>
#include <common/pipeline/ODDetection.h>

using namespace od;
using namespace std;

int main(int argc, char **argv)
{
	od::g2d::ODConvClassifier *cub = new od::g2d::ODConvClassifier();
	//TODO: Add usage example doumentation here	
	if (argc < 3){ 
		std::cout << "usage: ./build/examples/classification/cub/classify_cub" \
				<< " weight_file" <<"image" << std::endl;
		return 1;
	}
	else{
		string list_test(argv[2]);
		string model_def = "examples/classification/cub/deploy.prototxt";
		string weight_file(argv[1]);
		string mean_file = "examples/classification/cub/data/cub_mean.binaryproto";
		cub->initClassifier(model_def, weight_file);
		cub->setMeanFromFile(mean_file);
		int top_k = 3;
		float acc = cub->test("examples/classification/cub/data/images",list_test,top_k);
                std::cout << "Top " << top_k << " for list of images in "\
			<< list_test << " is: " << acc << std::endl;
	}
	return 0;
}
