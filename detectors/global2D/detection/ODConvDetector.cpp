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
#include "ODConvDetector.h"
using namespace caffe;

namespace od
{
	namespace g2d
	{

		void ODConvDetector::ODConvDetector::setConfig(std::string file_){
			API::Set_Config(file_);
		}

		void ODConvDetector::initDetector(std::string model_def, std::string model_weight){
			#ifdef WITH_GPU
			Caffe::set_mode(Caffe::GPU);
			#else
			Caffe::set_mode(Caffe::CPU);
			#endif
			//initiate the detector model
			detector = new API::Detector(model_def, model_weight);
		}

		std::vector<ODDetection2D*> ODConvDetector::detection(ODSceneImage *scene){
			cv::Mat image = scene->getCVImage();
			std::vector<caffe::Frcnn::BBox<float> > results;
			detector->predict(image, results);
			std::cout << "There are " << results.size() << " objects in picture." << std::endl;
			int num_objects = results.size();
			std::string label;
			double confidence_;
			std::vector<ODDetection2D*> detections;
			for (size_t obj = 0; obj < results.size(); obj++) {
				ODDetection2D *det = new ODDetection2D();
				label = caffe::Frcnn::GetClassName(caffe::Frcnn::LoadVocClass(),results[obj].id);
				det->setId(label);
				confidence_ = results[obj].confidence;
				det->setConfidence(confidence_);
				cv::Rect box(results[obj].Point[0], results[obj].Point[1], results[obj].Point[2] - results[obj].Point[0],
						results[obj].Point[3] - results[obj].Point[1] );
				det->setBoundingBox(box);
				detections.push_back(det);

			}
			return detections;
		}

		void ODConvDetector::printConfig(){
			caffe::Frcnn::FrcnnParam::print_param();
		}
	}
}
