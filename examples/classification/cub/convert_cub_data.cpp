#include <datasets/ODDatasetList.h>
#include<string>
//#include <common/pipeline/ObjectDetector.h>
//#include <common/pipeline/ODDetection.h>


using namespace od;
using namespace std;

int main(int argc, char **argv)
{
	//	string root_path(argv[1]);
	string root_path = "examples/classification/cub/data/";
	od::DatasetList *data = new od::DatasetList(root_path+"images/");
	string train_list = root_path + "train.txt";
	string test_list = root_path + "test.txt";

	string train_data_loc = root_path + "train_lmdb";
	string mean_image_loc = root_path + "cub_mean.binaryproto";
	string test_data_loc = root_path + "test_lmdb";
	//setting shuffling true for train data
	data->convert_dataset(train_list, train_data_loc, true, false, 256, 256);
	std::cout << "Convered Train data in lmdb at: " << train_data_loc << std::endl;
	
	data->compute_mean_image(train_data_loc, mean_image_loc);
        std::cout << "Mean Image from " << train_data_loc << " is stored at "
                << mean_image_loc << std::endl;


	// setting shuffling false for test data
	data->convert_dataset(test_list, test_data_loc, false, false, 256, 256);
	std::cout << "Convered Test data in lmdb at: " << test_data_loc << std::endl;
	return 0;
}


