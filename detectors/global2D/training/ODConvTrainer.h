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
				void setSolverParametersFromFile(std::string solver_location);
				void setSolverParameters(const std::string net,
                        			const float base_lr, const std::string lr_policy,const int max_iter,
                        			const int snapshot, const std::string snapshot_prefix);
				
				void getSolverParameters();
				void startTraining();
				void fineTuning(std::string weight_file_location);
				void resumeTraining(std::string solver_state_location);

			private:
				caffe::SolverParameter solver_param;
				
		};
	}
}


#endif //OPENDETECTION_ODCONVTRAINER_H
