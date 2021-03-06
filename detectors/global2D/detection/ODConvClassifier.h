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

#ifndef OPENDETECTION_ODCONVCLASSIFIER_H
#define OPENDETECTION_ODCONVCLASSIFIER_H

#include "ODClassifier.h"
#include <common/pipeline/ODDetection.h>

#include <caffe/caffe.hpp>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <algorithm>
#include <iosfwd>
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include<tuple> // for tuple>
#include <boost/assert.hpp>


namespace od
{
	namespace g2d
	{
		class ODConvClassifier: public ODClassifier
		{
			public:
				ODConvClassifier();
				virtual ~ODConvClassifier(){ }

				void init(){}

				//				void initTrainer(ODConvTrainer &trainer);

				void initClassifier(const std::string model_def_, const std::string model_weights_);	
				void setMeanFromFile(const std::string& mean_file);

				void setMeanFromArray(std::vector<float> &means);

				cv::Mat PreProcess(cv::Mat& img);      

				//				int train();

				ODDetections *classify (ODSceneImage *scene, int top=5);
				float test(const std::string root_path, const std::string label_file, int top=3);


			private:
				static std::vector<int> Argmax(const std::vector<float>& v, int N);
				static bool PairCompare(const std::pair<float, int>& lhs,
						const std::pair<float, int>& rhs);
				caffe::shared_ptr<caffe::Net<float> > net_;  
				//				ODConvTrainer trainer_*;
				cv::Size input_geometry_;
				int num_channels_;
				cv::Mat mean_;
		};
	}
}
#endif //OPENDETECTION_ODCONVCLASSIFIER_H
