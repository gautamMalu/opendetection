#include "ODConvTrainer.h"

using namespace caffe;
using namespace std;
using namespace cv;

namespace od
{
	namespace g2d
	{
		int ODConvTrainer::train()
		{
			return 1;
		}
		
		void ODConvTrainer::setSolverLocation(string location)
		{
			ODConvTrainer::solverLocation = location;	
		}
		
		std::string ODConvTrainer::getSolverLocation()
		{
			return solverLocation;
		}

		
		void ODConvTrainer::startTraining()
		{
		  caffe::SolverParameter solver_param;
		  caffe::ReadSolverParamsFromTextFileOrDie(solverLocation, &solver_param);
		  boost::shared_ptr<caffe::Solver<float> > solver(caffe::SolverRegistry<float>::CreateSolver(solver_param));
		  solver->Solve();
		}

		void ODConvTrainer::resumeTraining(std::string solver_state_location)
		{
		  caffe::SolverParameter solver_param;
                  caffe::ReadSolverParamsFromTextFileOrDie(solverLocation, &solver_param);
                  boost::shared_ptr<caffe::Solver<float> > solver(caffe::SolverRegistry<float>::CreateSolver(solver_param));
		  
		  const char * restore_file = solver_state_location.c_str();
    		  solver->Restore(restore_file);
		  solver->Solve();
		}
		//TODO: Implement Finetuning Method
		void ODConvTrainer::fineTuning(std::string weight_file_location)
		{
			std::cout << "Not Implemented Yet" <<std::endl;
		}	

		
	}
}
