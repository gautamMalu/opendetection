#include "common.h"
#include <google/protobuf/text_format.h>
#include <lmdb.h>
#include <stdint.h>
#include <sys/stat.h>

using namespace caffe;
using boost::scoped_ptr;
using std::string;


class MNISTDataset{
	public:
		MNISTDataset(const string& db_path, const string& db_backend = "lmdb"){
			std::string ROOT_PATH = db_path;
			std::string BACKEND  = db_backend;
			std::string train_images = "train-images-idx3-ubyte";
			std::string train_labels = "train-labels-idx1-ubyte";
			std::string test_images = "t10k-images-idx3-ubyte";
			std::string test_labels = "t10k-labels-idx1-ubyte";
		}

		void getTrainingData(const string& storageLocation){
			/* Generate training data in db_backend to be used in 
			   training by caffe*/
		}

		void getTestingData(const string& storageLocation){
			/* Generate testing data in db_backend to be used in 
			   training by caffe*/
		}
	protected:
		uint32_t swap_endian(uint32_t val) {
			val = ((val << 8) & 0xFF00FF00) | ((val >> 8) & 0xFF00FF);
			return (val << 16) | (val >> 16);
		}
		void convert_dataset(const char* image_filename, const char* label_filename,
        		const char* db_path, const string& db_backend){
		/* Actual Magic happens here*/
		}
};
