#include <detectors/global2D/detection/ODConvClassifier.h>
#include <common/pipeline/ODScene.h>
#include <common/pipeline/ODDetection.h>

using namespace od;
using namespace std;

int main(int argc, char **argv)
{
	od::g2d::ODConvClassifier *mnist = new od::g2d::ODConvClassifier();
	//TODO: Add usage example doumentation here	
	if (argc < 2){ 
		std::cout << "please provide an image" << std::endl;
		return 1;
	}
	else{
		string img_src(argv[1]);
		string model_def = "examples/mnist/lenet.prototxt";
		string weight_file = "examples/mnist/lenet_iter_10000.caffemodel";
		string mean_file = "examples/mnist/mnist_png/mnist_mean.binaryproto";
		mnist->initClassifier(model_def, weight_file);
		mnist->setMeanFromFile(mean_file);
		od::ODSceneImage *img = new od::ODSceneImage(img_src);
		std::vector<ODClassification2D*> labels = mnist->classify(img,1);
		cout <<  "label for " << img_src << " is "<<  labels[0]->getLabel() << " with confidence level of  " << labels[0]->getConfidence() << endl;	
	}
	return 0;
}
