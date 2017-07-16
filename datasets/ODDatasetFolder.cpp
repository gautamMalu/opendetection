#include "ODDatasetFolder.h"

using namespace caffe;
using boost::scoped_ptr;
using namespace std;
namespace fs = boost::filesystem;


DatasetFolder::DatasetFolder(const string& db_path, const string& db_backend){
	string ROOT_PATH = db_path;
	string BACKEND  = db_backend;
}

void DatasetFolder::convert_dataset(const string& input_folder, const string& storageLocation, bool shuffling, 
		bool gray, int resize_width, int resize_height, 
		bool check_size, bool encoded, const string& encode_type){

	std::vector<std::pair<std::string, int> > lines;
	std::string root_path = input_folder;
	read_directory(root_path, labels);
	int n = labels.size();
	vector<string> filenames;
	int i,j;
	string label_path_str;

	for(i=0;i < n;i++)
	{
		fs::path label_path(labels[i]);
		//read each label's directory
		label_path_str = label_path.string();
		read_directory(label_path_str, filenames);
		//make a pair
		for (j=0; j< filenames.size(); j++) {
			lines.push_back(std::make_pair(filenames[j], i));
		}
		filenames.clear();
	}

	const bool is_color = !gray;

	if (shuffling) {
		// randomly shuffle data
		std::cout << "Shuffling data" << std::endl;
		shuffle(lines.begin(), lines.end());
	}
	std::cout << "A total of " << lines.size() << " images." << std::endl;
	if (encode_type.size() && !encoded)
		std::cout << "encode_type specified, assuming encoded=true." << std::endl;
	// Create new DB

	scoped_ptr<db::DB> db(db::GetDB(BACKEND));
	db->Open(storageLocation.c_str(), db::NEW);
	scoped_ptr<db::Transaction> txn(db->NewTransaction());

	Datum datum;
	int count = 0;
	int data_size = 0;	
	bool data_size_initialized = false;

	for(int line_id = 0; line_id < lines.size(); ++line_id) {
		bool status;
		std::string enc = encode_type;
		if (encoded && !enc.size()) {
			// Guess the encoding type from the file name
			string fn = lines[line_id].first;
			size_t p = fn.rfind('.');
			if ( p == fn.npos )
				std::cout << "Failed to guess the encoding of '" << fn << "'"<< std::endl;
			enc = fn.substr(p);
			std::transform(enc.begin(), enc.end(), enc.begin(), ::tolower);
		}
		status = ReadImageToDatum(lines[line_id].first,
				lines[line_id].second, resize_height, resize_width, is_color,enc, &datum);
		if (status == false) continue;
		if (check_size) {
			if (!data_size_initialized) {
				data_size = datum.channels() * datum.height() * datum.width();
				data_size_initialized = true;
			} else {
				const std::string& data = datum.data();
				/*	CHECK_EQ(data.size(), data_size) << "Incorrect data field size "
					<< data.size();*/
			}
		}
		// sequential
		string key_str = caffe::format_int(line_id, 8) + "_" + lines[line_id].first;

		// Put in db
		string out;
		/*CHECK(*/datum.SerializeToString(&out);//);
		txn->Put(key_str, out);

		if (++count % 1000 == 0) {
			// Commit db
			txn->Commit();
			txn.reset(db->NewTransaction());
			std::cout << "Processed " << count << " files."<<std::endl;
		}
	}


}

void DatasetFolder::compute_mean_image(const string& INPUT_DB, const string& OUTPUT_FILE){
	/* Compute mean image with given db file and save at given output location*/
	scoped_ptr<db::DB> db(db::GetDB(BACKEND));
	db->Open(INPUT_DB.c_str(), db::READ);
	scoped_ptr<db::Cursor> cursor(db->NewCursor());

	BlobProto sum_blob;
	int count = 0;
	// load first datum
	Datum datum;
	datum.ParseFromString(cursor->value());

	if (DecodeDatumNative(&datum)) {
		std::cout << "Decoding Datum" <<std::endl;
	}
	sum_blob.set_num(1);
	sum_blob.set_channels(datum.channels());
	sum_blob.set_height(datum.height());
	sum_blob.set_width(datum.width());
	const int data_size = datum.channels() * datum.height() * datum.width();
	int size_in_datum = std::max<int>(datum.data().size(),
			datum.float_data_size());
	for (int i = 0; i < size_in_datum; ++i) {
		sum_blob.add_data(0.);
	}
	std::cout << "Starting iteration" <<std::endl;
	while (cursor->valid()) {
		Datum datum;
		datum.ParseFromString(cursor->value());
		DecodeDatumNative(&datum);

		const std::string& data = datum.data();
		size_in_datum = std::max<int>(datum.data().size(),
				datum.float_data_size());
		/*CHECK_EQ(size_in_datum, data_size) << "Incorrect data field size " <<
		  size_in_datum;*/
		if (data.size() != 0) {
			/*CHECK_EQ(data.size(), size_in_datum);*/
			for (int i = 0; i < size_in_datum; ++i) {
				sum_blob.set_data(i, sum_blob.data(i) + (uint8_t)data[i]);
			}
		} else {
			/*CHECK_EQ(datum.float_data_size(), size_in_datum);*/
			for (int i = 0; i < size_in_datum; ++i) {
				sum_blob.set_data(i, sum_blob.data(i) +
						static_cast<float>(datum.float_data(i)));
			}
		}
		++count;
		if (count % 10000 == 0) {
			std::cout << "Processed " << count << " files." <<std::endl;
		}
		cursor->Next();
	}

	if (count % 10000 != 0) {
		std::cout << "Processed " << count << " files."<<std::endl;
	}
	for (int i = 0; i < sum_blob.data_size(); ++i) {
		sum_blob.set_data(i, sum_blob.data(i) / count);
	}
	// Write to disk

	std::cout << "Write to " << OUTPUT_FILE <<std::endl;
	WriteProtoToBinaryFile(sum_blob, OUTPUT_FILE.c_str());

	const int channels = sum_blob.channels();
	const int dim = sum_blob.height() * sum_blob.width();
	std::vector<float> mean_values(channels, 0.0);
	std::cout << "Number of channels: " << channels << std::endl;
	for (int c = 0; c < channels; ++c) {
		for (int i = 0; i < dim; ++i) {
			mean_values[c] += sum_blob.data(dim * c + i);
		}
		std::cout << "mean_value channel [" << c << "]: " << mean_values[c] / dim << std::endl;
	}
}


void DatasetFolder::get_labels()
{	//get labels information
	if(!labels.empty())
	{
		for(int i=0;i<labels.size();i++){
			std::cout << labels[i] << " " << i << std::endl;
		}
	}
	else
		std::cout << "labels vector is empty" << " first run convert_dataset." << std::endl;
}

void DatasetFolder::read_directory(string& name, vector<string>& v)
{
	fs::path p(name);
	fs::directory_iterator start(p);
	fs::directory_iterator end;
	std::transform(start, end, std::back_inserter(v), [](fs::path p) { return p.string();});
}

