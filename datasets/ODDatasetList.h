#ifndef ODDATASETLIST_H
#define ODDATASETLIST_H

#include "common.h"
#include <google/protobuf/text_format.h>
#include <lmdb.h>
#include <stdint.h>
#include <sys/stat.h>

using namespace caffe;
using boost::scoped_ptr;
using std::string;


class DatasetList{
	public:
		DatasetList(const std::string& db_path, const std::string& db_backend = "lmdb");
		void compute_mean_image(const string& INPUT_DB, const string& OUTPUT_FILE);
		void convert_dataset(const string& list, const string& storageLocation, bool shuffling = false, 
			bool gray = false, int resize_width = 0, int resize_height = 0, 
			bool check_size = false, bool encoded = false, const string& encode_type= "");
	private:
		 std::string ROOT_PATH;
        	 std::string BACKEND;

};
#endif //ODDATASETLIST_H
