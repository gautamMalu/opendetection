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

#include "ODClassifier.h"


namespace od
{
	namespace g2d
	{


		ODClassifier::ODClassifier(DetectionMethod detection_method){
			method_ = detection_method;
		}

		void ODClassifier::init(){}

		void ODClassifier::initTrainer(ODTrainer *trainer){}

		void ODClassifier::initDetector(){}

		int ODClassifier::train(){}

		ODDetections * ODClassifier::classify(ODSceneImage *scene){}

		DetectionMethod const & ODClassifier::getDetectiontype() const
		{
			return method_;
		}

		void ODClassifier::setDetectionMethod(DetectionMethod const &detection_method_)
		{
			this->method_ = detection_method_;
		}


	}
}