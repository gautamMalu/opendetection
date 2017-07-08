#include "ODDatasetCIFAR.h"

using namespace caffe;
using boost::scoped_ptr;
using std::string;
namespace db = caffe::db;

CIFARDataset::CIFARDataset(const string& root_path, const string& db_backend){
	std::string ROOT_PATH = root_path;
	// removing the "/" if present
	if (boost::ends_with(ROOT_PATH, "/")){
		ROOT_PATH.pop_back();	
	}
	std::string BACKEND  = db_backend;
}

void CIFARDataset::getTrainingData(const string& storageLocation){

	/* Generate training data in db_backend to be used in 
	   training by caffe*/
	std::string input_folder = ROOT_PATH;
	std::string output_folder = storageLocation;
	std::string db_type = BACKEND;
	scoped_ptr<caffe::db::DB> train_db(caffe::db::GetDB(db_type));
	train_db->Open(output_folder + "/cifar10_train_" + db_type, caffe::db::NEW);
	scoped_ptr<caffe::db::Transaction> txn(train_db->NewTransaction());
	// Data buffer
	int label;
	char str_buffer[kCIFARImageNBytes];
	Datum datum;
	datum.set_channels(3);
	datum.set_height(kCIFARSize);
	datum.set_width(kCIFARSize);

	std::cout << "Writing Training data" << std::endl;
	for (int fileid = 0; fileid < kCIFARTrainBatches; ++fileid) {
		// Open files
		std::cout << "Training Batch " << fileid + 1 << std::endl;
		string batchFileName = input_folder + "/data_batch_"
			+ caffe::format_int(fileid+1) + ".bin";
		std::ifstream data_file(batchFileName.c_str(),
				std::ios::in | std::ios::binary);
		//CHECK(data_file) << "Unable to open train file #" << fileid + 1;
		for (int itemid = 0; itemid < kCIFARBatchSize; ++itemid) {
			read_image(&data_file, &label, str_buffer);
			datum.set_label(label);
			datum.set_data(str_buffer, kCIFARImageNBytes);
			string out;
			/*CHECK(*/datum.SerializeToString(&out);//);
			txn->Put(caffe::format_int(fileid * kCIFARBatchSize + itemid, 5), out);
		}
	}
	txn->Commit();
	train_db->Close();

}

void CIFARDataset::getTestingData(const string& storageLocation){
	/* Generate testing data in db_backend to be used in 
	   training by caffe*/
	std::cout << "Writing Testing data" << std::endl;
	std::string input_folder = ROOT_PATH;
	std::string output_folder = storageLocation;
	std::string db_type = BACKEND;

	scoped_ptr<caffe::db::DB> test_db(caffe::db::GetDB(db_type));
	test_db->Open(output_folder + "/cifar10_test_" + db_type, caffe::db::NEW);
	scoped_ptr<caffe::db::Transaction> txn(test_db->NewTransaction());

	//txn.reset(test_db->NewTransaction());
	// Data buffer
	int label;
	char str_buffer[kCIFARImageNBytes];
	Datum datum;
	datum.set_channels(3);
	datum.set_height(kCIFARSize);
	datum.set_width(kCIFARSize);

	// Open files
	std::ifstream data_file((input_folder + "/test_batch.bin").c_str(),
			std::ios::in | std::ios::binary);
	//CHECK(data_file) << "Unable to open test file.";
	for (int itemid = 0; itemid < kCIFARBatchSize; ++itemid) {
		read_image(&data_file, &label, str_buffer);
		datum.set_label(label);
		datum.set_data(str_buffer, kCIFARImageNBytes);
		string out;
		/*CHECK(*/datum.SerializeToString(&out);//);
		txn->Put(caffe::format_int(itemid, 5), out);
	}
	txn->Commit();
	test_db->Close();
}



void CIFARDataset::read_image(std::ifstream* file, int* label, char* buffer) {
	char label_char;
	file->read(&label_char, 1);
	*label = label_char;
	file->read(buffer, kCIFARImageNBytes);
	return;
}
