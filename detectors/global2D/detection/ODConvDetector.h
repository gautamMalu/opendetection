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
// Created by sarkar on 15.07.15.
//

#ifndef OPENDETECTION_ODCONVDETECTOR_H
#define OPENDETECTION_ODCONVDETECTOR_H

#include "common/pipeline/ODDetector.h"
#include "common/pipeline/ODScene.h"
#include "common/utils/utils.h"

#include <iostream>
#include <caffe/caffe.hpp>
#include <opencv2/opencv.hpp>

#include <gflags/gflags.h>
#include <glog/logging.h>
#include "boost/algorithm/string.hpp"
#include "caffe/util/benchmark.hpp"
#include "caffe/util/signal_handler.h"
#include "caffe/FRCNN/util/frcnn_vis.hpp"
#include "api/api.hpp"


namespace od
{
	namespace g2d
	{
		class ODConvDetector : public ODDetector2D
		{
			public:

				ODConvDetector(std::string const &trained_data_location_ = ""):ODDetector2D(trained_data_location_){}
                                virtual ~ODConvDetector(){ }

				void init(){}
				void setConfig(std::string file_);

				void initDetector(std::string model_def, std::string model_weight);

				std::vector<ODDetection2D*> detection(ODSceneImage *scene);
				ODDetections *detect(ODSceneImage *scene){}
				ODDetections2D* detectOmni(ODSceneImage *scene){}

				int detect(ODScene *scene, std::vector<ODDetection *> &detections)
				{ }
				void printConfig();

			protected:
				//properteis
				API::Detector* detector;
		};
	}
}
#endif //OPENDETECTION_ODCONVDETECTOR_H
