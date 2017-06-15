#ifndef OPENDETECTION_ODCONVTRAINER_H
#define OPENDETECTION_ODCONVTRAINER_H


#include "common/pipeline/ODTrainer.h"
#include "common/utils/utils.h"
#include <opencv2/opencv.hpp>

#include <cstring>
#include <cstdlib>
#include <vector>

#include <string>
#include <iostream>
#include <stdio.h>
#include <assert.h>     /* assert */
#include "caffe/caffe.hpp"
#include "caffe/util/io.hpp"
#include "caffe/blob.hpp"
#include "caffe/solver.hpp"
#include "caffe/sgd_solvers.hpp"

using namespace caffe;
using namespace std;
using namespace cv;

namespace od
{
	namespace g2d
	{
		class ODConvTrainer : public ODTrainer
		{	
			public:
				ODConvTrainer(std::string const &training_input_location_ = "", std::string const &trained_data_location_ = ""):ODTrainer(training_input_location_, trained_data_location_){}
			
				int train();  
				void init(){} 
				void setSolverLocation(std::string location);
				std::string getSolverLocation();
				void startTraining();
				void fineTuning(std::string weight_file_location);
				void resumeTraining(std::string solver_state_location);

			private:
				std::string solverLocation;
		};
	}
}


#endif //OPENDETECTION_ODCONVTRAINER_H
