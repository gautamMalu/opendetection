#include "ODDatasetMNIST.h"

using namespace caffe;
using boost::scoped_ptr;
using std::string;


MNISTDataset::MNISTDataset(const string& root_path, const string& db_backend){
	//Adding "/" to the path
	if (boost::ends_with(root_path, "/")){
		std::string ROOT_PATH = root_path;
	}
	else{
		std::string ROOT_PATH = root_path + "/";
	}
	std::string ROOT_PATH = root_path;
	std::string BACKEND  = db_backend;
	std::string train_images = "train-images-idx3-ubyte";
	std::string train_labels = "train-labels-idx1-ubyte";
	std::string test_images = "t10k-images-idx3-ubyte";
	std::string test_labels = "t10k-labels-idx1-ubyte";
}

void MNISTDataset::getTrainingData(const string& storageLocation){
	/* Generate training data in db_backend to be used in 
	   training by caffe*/
	const char* image_filename = (ROOT_PATH + train_images).c_str();
	const char* label_filename = (ROOT_PATH + train_labels).c_str();
	const char* db_path = storageLocation.c_str();
	const std::string db_backend = BACKEND;

	convert_dataset(image_filename, label_filename, db_path, db_backend);

}

void MNISTDataset::getTestingData(const string& storageLocation){
	/* Generate testing data in db_backend to be used in 
	   training by caffe*/
	const char* image_filename = (ROOT_PATH + test_images).c_str();
	const char* label_filename = (ROOT_PATH + test_labels).c_str();
	const char* db_path = storageLocation.c_str();
	const std::string db_backend = BACKEND;
	convert_dataset(image_filename, label_filename, db_path, db_backend);

}

uint32_t MNISTDataset::swap_endian(uint32_t val) {
	val = ((val << 8) & 0xFF00FF00) | ((val >> 8) & 0xFF00FF);
	return (val << 16) | (val >> 16);
}

void MNISTDataset::convert_dataset(const char* image_filename, const char* label_filename,
		const char* db_path, const string& db_backend){
	// Open files
	std::ifstream image_file(image_filename, std::ios::in | std::ios::binary);
	std::ifstream label_file(label_filename, std::ios::in | std::ios::binary);
	//TODO: Replace CHECK flags of glog
	//CHECK(image_file) << "Unable to open file " << image_filename;
	//CHECK(label_file) << "Unable to open file " << label_filename;
	// Read the magic and the meta data
	uint32_t magic;
	uint32_t num_items;
	uint32_t num_labels;
	uint32_t rows;
	uint32_t cols;

	image_file.read(reinterpret_cast<char*>(&magic), 4);
	magic = swap_endian(magic);
	assert(("Incorrect image file magic.", magic==2051));
	label_file.read(reinterpret_cast<char*>(&magic), 4);
	magic = swap_endian(magic);
	assert(("Incorrect label file magic.", magic==2049));
	image_file.read(reinterpret_cast<char*>(&num_items), 4);
	num_items = swap_endian(num_items);
	label_file.read(reinterpret_cast<char*>(&num_labels), 4);
	num_labels = swap_endian(num_labels);
	assert(("Number of item should be equal to number of labels",num_items==num_labels));
	image_file.read(reinterpret_cast<char*>(&rows), 4);
	rows = swap_endian(rows);
	image_file.read(reinterpret_cast<char*>(&cols), 4);
	cols = swap_endian(cols);


	scoped_ptr<db::DB> db(db::GetDB(db_backend));
	db->Open(db_path, db::NEW);
	scoped_ptr<db::Transaction> txn(db->NewTransaction());

	// Storing to db
	char label;
	char* pixels = new char[rows * cols];
	int count = 0;
	string value;

	Datum datum;
	datum.set_channels(1);
	datum.set_height(rows);
	datum.set_width(cols);
	std::cout << "A total of " << num_items << " items." << std::endl;
	std::cout << "Rows: " << rows << " Cols: " << cols << std::endl;
	for (int item_id = 0; item_id < num_items; ++item_id) {
		image_file.read(pixels, rows * cols);
		label_file.read(&label, 1);
		datum.set_data(pixels, rows*cols);
		datum.set_label(label);
		string key_str = caffe::format_int(item_id, 8);
		datum.SerializeToString(&value);

		txn->Put(key_str, value);

		if (++count % 1000 == 0) {
			txn->Commit();
		}
	}
	// write the last batch
	if (count % 1000 != 0) {
		txn->Commit();
	}
	std::cout << "Processed " << count << " files." << std::endl;
	delete[] pixels;
	db->Close();
}
