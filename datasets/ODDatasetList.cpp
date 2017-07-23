#include "ODDatasetList.h"

using namespace caffe;
using boost::scoped_ptr;
using std::string;


namespace od
{
	DatasetList::DatasetList(const string& db_path, const string& db_backend){
		ROOT_PATH = db_path;
		BACKEND  = db_backend;
	}

	void DatasetList::convert_dataset(const string& list, const string& storageLocation, bool shuffling, 
			bool gray, int resize_width, int resize_height, 
			bool check_size, bool encoded, const string& encode_type){

		const bool is_color = !gray;
		std::ifstream infile(list.c_str());
		std::vector<std::pair<std::string, int> > lines;
		std::string line;
		size_t pos;
		int label;
		while (std::getline(infile, line)) {
			pos = line.find_last_of(' ');
			label = atoi(line.substr(pos + 1).c_str());
			lines.push_back(std::make_pair(line.substr(0, pos), label));
		}
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
		std::string root_folder= ROOT_PATH;
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
			status = ReadImageToDatum(root_folder + lines[line_id].first,
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

	void DatasetList::compute_mean_image(const string& INPUT_DB, const string& OUTPUT_FILE){
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
}
