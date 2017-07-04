#include "common.h"
#include <google/protobuf/text_format.h>
#include <lmdb.h>
#include <stdint.h>
#include <sys/stat.h>

using namespace caffe;
using boost::scoped_ptr;
using std::string;


class DatasetFolder{
	public:
		DatasetFolder(const string& db_path, const string& db_backend = "lmdb"){
			std::string ROOT_PATH = db_path;
			std::string BACKEND  = db_backend;
		}

		void convert_dataset(bool shuffle = false, 
			bool gray = false, int resize_width = 0, int resize_height = 0, 
			bool check_size = false, bool encoded = false, const string& ecode_type= ""){
		/* Actual Magic happens here*/
		}
		
		void compute_mean_image(const string& INPUT_DB, const string& OUTPUT_FILE){
		/* Compute mean image with given db file and save at given output location*/
		}
		
		void get_labels(const string& labels_xml_location)
		{/* return an xml file with labels info*/
		}
		
};
