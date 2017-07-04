#include "common.h"
#include <google/protobuf/text_format.h>
#include <stdint.h>


using namespace caffe;
using boost::scoped_ptr;
using std::string;
namespace db = caffe::db;


class CIFARDataset{
	public:
		CIFARDataset(const string& db_path, const string& db_backend = "lmdb"){
			std::string ROOT_PATH = db_path;
			std::string BACKEND  = db_backend;
			std::string test_data = "/test_batch.bin";
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
		void read_image(std::ifstream* file, int* label, char* buffer) {
  			char label_char;
			const int kCIFARImageNBytes = 3072;
			file->read(&label_char, 1);
  			*label = label_char;
  			file->read(buffer, kCIFARImageNBytes);
  			return;
			}
	
		void convert_dataset(const string& input_folder, const string& output_folder,
    			const string& db_type){
		/* Actual Magic happens here*/
		}
};
