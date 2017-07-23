#include "datasets/ODDatasetFolder.h"
//#include "detectors/global2D/detection/ODConvClassification.h"
#include<string>
#include "common/pipeline/ObjectDetector.h"
#include "common/pipeline/ODDetection.h"


using namespace od;
using namespace std;

int main(int argc, char **argv)
{
	od::DatasetFolder *mnist_train = new od::DatasetFolder("","lmdb");
//	od::DatasetFolder *mnist_test = new od::DatasetFolder("","lmdb");

	//TODO: Add usage example doumentation here	
	if (argc < 2){ 
		std::cout << "please provide root path for images" << std::endl;
		return 1;
	}
	else{
		string root_path(argv[1]);
		cout << root_path << endl;
		string train_images_loc = root_path + "training";
		string test_images_loc = root_path + "testing";
		string train_data_loc = root_path + "train_lmdb"; //train data location
		string test_data_loc = root_path + "test_lmdb";
		string mean_image_loc = root_path + "mnist_mean.binaryproto";

		mnist_train->convert_dataset(train_images_loc,train_data_loc,true,true,0,0,false,false,"");
		mnist_train->convert_dataset(test_images_loc,test_data_loc,true,true,0,0,false,false,"");
		mnist_train->compute_mean_image(train_data_loc, mean_image_loc);

		mnist_train->printLabels();
		return 0;
	}

	return 0;
}

