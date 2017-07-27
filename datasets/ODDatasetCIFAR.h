#ifndef ODCIFARDATASET_H
#define ODCIFARDATASET_H


#include "common.h"
#include <google/protobuf/text_format.h>
#include <boost/algorithm/string/predicate.hpp>
#include <stdint.h>

namespace od
{
	class CIFARDataset{
		public:
			CIFARDataset(const std::string& root_path, const std::string& db_backend = "lmdb");
			virtual ~CIFARDataset(){}
			void getTrainingData(const std::string& storageLocation);
			void getTestingData(const std::string& storageLocation);

		private:
			std::string ROOT_PATH; 
			std::string BACKEND; 
			std::string test_data;

			const int kCIFARSize = 32;
			const int kCIFARImageNBytes = 3072;
			const int kCIFARBatchSize = 10000;
			const int kCIFARTrainBatches = 5;

		protected:
			void read_image(std::ifstream* file, int* label, char* buffer);
	};
}
#endif //ODCIFARDATASET_H
