#ifndef ODMNISTDATASET_H
#define ODMNISTDATASET_H

#include "common.h"
#include <google/protobuf/text_format.h>
#include <lmdb.h>
#include <stdint.h>
#include <sys/stat.h>
#include <boost/algorithm/string/predicate.hpp>

namespace od
{
	class MNISTDataset{
		public: 
			MNISTDataset(const std::string& root_path, const std::string& db_backend = "lmdb");
			virtual ~MNISTDataset(){}
			void getTrainingData(const std::string& storageLocation);
			void getTestingData(const std::string& storageLocation);

		private:
			std::string ROOT_PATH;
			std::string BACKEND;
			std::string train_images;
			std::string train_labels;
			std::string test_images;
			std::string test_labels;

		protected:
			uint32_t swap_endian(uint32_t val);
			void convert_dataset(const char* image_filename, const char* label_filename,
					const char* db_path, const std::string& db_backend);
	};
}
#endif //ODMNISTDATASET_H

