/*
   Copyright (c) 2015, Kripasindhu Sarkar
   All rights reserved.

   Redistribution and use in source and binary forms, with or without
   modification, are permitted provided that the following conditions are met:
 * Redistributions of source code must retain the above copyright
 notice, this list of conditions and the following disclaimer.
 * Redistributions in binary form must reproduce the above copyright
 notice, this list of conditions and the following disclaimer in the
 documentation and/or other materials provided with the distribution.
 * Neither the name of the copyright holder(s) nor the
 names of its contributors may be used to endorse or promote products
 derived from this software without specific prior written permission.

 THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 DISCLAIMED. IN NO EVENT SHALL COPYRIGHT HOLDER BE LIABLE FOR ANY
 DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *///
// Created by sarkar on 16.06.15.
//


#include "ODConvClassifier.h"

using namespace caffe; 
using std::string;

namespace od
{
	namespace g2d
	{
		ODConvClassifier::ODConvClassifier(){}


		void ODConvClassifier::initClassifier(const std::string model_def_, const std::string model_weights_){
		#ifdef WITH_GPU
			Caffe::set_mode(Caffe::GPU);
		#else
			Caffe::set_mode(Caffe::CPU);
		#endif
			net_.reset(new Net<float>(model_def_, TEST));
			net_->CopyTrainedLayersFrom(model_weights_);
			BOOST_ASSERT_MSG(net_->num_inputs()==1,"Network should have exactly one input.");
			BOOST_ASSERT_MSG(net_->num_outputs()==1,"Network should have exactly one output.");
			Blob<float>* input_layer = net_->input_blobs()[0];
			num_channels_ = input_layer->channels();
			BOOST_ASSERT_MSG((num_channels_ == 3 || num_channels_ == 1 ),
					"Input layer should have 1 or 3 channels.");
			input_geometry_ = cv::Size(input_layer->width(), input_layer->height());

		}	

		void ODConvClassifier::setMeanFromFile(const string& mean_file){
			BlobProto blob_proto;
			ReadProtoFromBinaryFileOrDie(mean_file.c_str(), &blob_proto);

			/* Convert from BlobProto to Blob<float> */
			Blob<float> mean_blob;
			mean_blob.FromProto(blob_proto);
			BOOST_ASSERT_MSG(mean_blob.channels()==num_channels_,
					"Number of channels of mean file doesn't match input layer.");

			/* The format of the mean file is planar 32-bit float BGR or grayscale. */
			std::vector<cv::Mat> channels;
			float* data = mean_blob.mutable_cpu_data();
			for (int i = 0; i < num_channels_; ++i) {
				/* Extract an individual channel. */
				cv::Mat channel(mean_blob.height(), mean_blob.width(), CV_32FC1, data);
				channels.push_back(channel);
				data += mean_blob.height() * mean_blob.width();
			}

			/* Merge the separate channels into a single image. */
			cv::Mat mean;
			cv::merge(channels, mean);

			/* Compute the global mean pixel value and create a mean image
			 * filled with this value. */
			cv::Scalar channel_mean = cv::mean(mean);
			mean_ = cv::Mat(input_geometry_, mean.type(), channel_mean);
		}

		void ODConvClassifier::setMeanFromArray(std::vector<float> &means){
			BOOST_ASSERT_MSG(means.size()==num_channels_,
					"Number of channels of mean vector file doesn't match input layer.");
			std::vector<cv::Mat> channels;
			for (int i =0; i < num_channels_;i++){
				cv::Mat channel(input_geometry_,CV_32FC1);
				channel = cv::Scalar(means[i]);
			}
			cv::merge(channels,mean_);
		}

		cv::Mat ODConvClassifier::PreProcess(cv::Mat& img){
			cv::Mat sample;
			if (img.channels() == 3 && num_channels_ == 1)
				cv::cvtColor(img, sample, cv::COLOR_BGR2GRAY);
			else if (img.channels() == 4 && num_channels_ == 1)
				cv::cvtColor(img, sample, cv::COLOR_BGRA2GRAY);
			else if (img.channels() == 4 && num_channels_ == 3)
				cv::cvtColor(img, sample, cv::COLOR_BGRA2BGR);
			else if (img.channels() == 1 && num_channels_ == 3)
				cv::cvtColor(img, sample, cv::COLOR_GRAY2BGR);
			else
				sample = img;

			cv::Mat sample_resized;
			if (sample.size() != input_geometry_)
				cv::resize(sample, sample_resized, input_geometry_);
			else
				sample_resized = sample;

			cv::Mat sample_float;
			if (num_channels_ == 3)
				sample_resized.convertTo(sample_float, CV_32FC3);
			else
				sample_resized.convertTo(sample_float, CV_32FC1);

			cv::Mat sample_normalized;
			if (!mean_.empty()){
				cv::subtract(sample_float, mean_, sample_normalized);
				return sample_normalized;
			}
			else{
				return sample_float;
			}

		}    

		std::vector<ODClassification2D*> ODConvClassifier::classify (ODSceneImage *scene, int top)
		{		
			cv::Mat img_input = scene->getCVImage();
			cv::Mat input = PreProcess(img_input);
			Blob<float>* input_layer = net_->input_blobs()[0];
			input_layer->Reshape(1, num_channels_,
					input_geometry_.height, input_geometry_.width);
			/* Forward dimension change to all layers. */
			net_->Reshape();
			std::vector<cv::Mat> input_channels;
			/* Wrap the input layer of the network in separate cv::Mat objects
			 * (one per channel). This way we save one memcpy operation and we
			 * don't need to rely on cudaMemcpy2D. The last preprocessing
			 * operation will write the separate channels directly to the input
			 * layer. */
			int width = input_layer->width();
			int height = input_layer->height();
			float* input_data = input_layer->mutable_cpu_data();
			for (int i = 0; i < input_layer->channels(); ++i) {
				cv::Mat channel(height, width, CV_32FC1, input_data);
				input_channels.push_back(channel);
				input_data += width * height;
			}
			/* This operation will write the separate BGR planes directly to the
			 * input layer of the network because it is wrapped by the cv::Mat
			 * objects in input_channels. */
			cv::split(input, input_channels);
			BOOST_ASSERT_MSG(reinterpret_cast<float*>(input_channels.at(0).data)
					== net_->input_blobs()[0]->cpu_data(), 
					"Input channels are not wrapping the input layer of the network.");

			net_->Forward();

			/* Copy the output layer to a std::vector */
			Blob<float>* output_layer = net_->output_blobs()[0];
			const float* begin = output_layer->cpu_data();
			const float* end = begin + output_layer->channels();
			std::vector<float> outputs_(begin, end);
			std::vector<int> topk = Argmax(outputs_,top);
			std::vector<ODClassification2D*> outputs;
			for (int i=0;i<top;i++)
			{
				ODClassification2D *output = new ODClassification2D( topk[i],double(outputs_[topk[i]]));
				outputs.push_back(output);
			} 
			return outputs;
		}

		/* Return the indices of the top N values of vector v. */
		std::vector<int> ODConvClassifier::Argmax(const std::vector<float>& v, int N) {
			std::vector<std::pair<float, int> > pairs;
			for (size_t i = 0; i < v.size(); ++i)
				pairs.push_back(std::make_pair(v[i], i));
			std::partial_sort(pairs.begin(), pairs.begin() + N, pairs.end(), PairCompare);

			std::vector<int> result;
			for (int i = 0; i < N; ++i)
				result.push_back(pairs[i].second);
			return result;
		}

		bool ODConvClassifier::PairCompare(const std::pair<float, int>& lhs,
				const std::pair<float, int>& rhs) {
			return lhs.first > rhs.first;
		}

		float ODConvClassifier::test(const string root_path, const string label_file, int top){

			std::ifstream infile(label_file.c_str());
			std::string line;
			size_t pos;
			od::ODSceneImage *img;
			std::vector<ODClassification2D*> labels;
			int predicted_label, ground_truth;
			std::string img_name,img_src;

			int num_samples = 0;
			int num_correct_samples=0;
			int i; //for indexing
			while (std::getline(infile, line)) {
				pos = line.find_last_of(' ');
				ground_truth = atoi(line.substr(pos + 1).c_str());
				img_name = line.substr(0, pos);
				img_src = root_path + '/' + img_name;
				img = new od::ODSceneImage(img_src);
				labels = classify(img, top);
				for (i=0; i<top; i++){
					if(ground_truth == labels[i]->getLabel()){
						num_correct_samples++;
						break;
					}
				}
				num_samples++;
			}

			std::cout << num_correct_samples << " out of " << num_samples << " are correctly predicted." << std::endl;
			float accuracy = float(num_correct_samples)/float(num_samples);
			return accuracy;
		}

	}
}
