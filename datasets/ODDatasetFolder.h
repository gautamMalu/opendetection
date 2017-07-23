#ifndef ODDATASETFOLDER_H
#define ODDATASETFOLDER_H

#include "common.h"
#include <google/protobuf/text_format.h>
#include <sys/stat.h>
#include "boost/filesystem/operations.hpp"
#include <boost/range/combine.hpp> // combine function
#include "boost/filesystem/path.hpp"
#include <boost/algorithm/string.hpp>
#include <boost/range/algorithm.hpp>
#include <iostream>

namespace od
{
	class DatasetFolder{
		public:
			DatasetFolder(const std::string& db_path, const std::string& db_backend = "lmdb");
			virtual ~DatasetFolder(){}
			void convert_dataset(const std::string& input_folder, const std::string& storageLocation, 
					bool shuffling = false, bool gray = false, int resize_width = 0, 
					int resize_height = 0, bool check_size = false, 
					bool encoded = false, const std::string& ecode_type= "");

			void compute_mean_image(const std::string& INPUT_DB, const std::string& OUTPUT_FILE);
			std::vector<std::string> getLabels();
			void printLabels();
		private:
			std::string ROOT_PATH;
			std::string BACKEND;
			std::vector <std::string> labels_;
			void read_directory(std::string& name, std::vector<std::string>& v);
	};
}
#endif //ODDATASETFOLDER_H
