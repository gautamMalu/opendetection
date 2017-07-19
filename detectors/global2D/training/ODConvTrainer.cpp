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

		void ODConvTrainer::setSolverParametersFromFile(std::string solver_location)
		{
			caffe::ReadSolverParamsFromTextFileOrDie(solver_location, &solver_param);	
		}
		
		void ODConvTrainer::setSolverParameters(const std::string net, 
			const float base_lr, const std::string lr_policy, const int max_iter, 
			const int snapshot, const std::string snapshot_prefix){

		int display = 40;
		int test_interval = snapshot; // test before each snapshot
		int test_iter = 40; // number of test iterations
		solver_param.set_display(display);
		solver_param.set_net(net);
		solver_param.set_base_lr(base_lr);
		solver_param.set_lr_policy(lr_policy);
		solver_param.set_max_iter(max_iter);
		solver_param.set_snapshot(snapshot);
		solver_param.set_snapshot_prefix(snapshot_prefix);
		solver_param.set_test_interval(test_interval);
		solver_param.add_test_iter(test_iter);
		}

		void ODConvTrainer::getSolverParameters()
		{
			std::cout << solver_param.DebugString() << std::endl;
		}


		void ODConvTrainer::startTraining()
		{
		//	caffe::SolverParameter solver_param;
//			caffe::ReadSolverParamsFromTextFileOrDie(solverLocation, &solver_param);
//			solver_param.set_net("changed net");
			std::cout << solver_param.DebugString() << std::endl;
			boost::shared_ptr<caffe::Solver<float> > solver(caffe::SolverRegistry<float>::CreateSolver(solver_param));
			solver->Solve();
		}

		void ODConvTrainer::resumeTraining(std::string solver_state_location)
		{
		//	caffe::SolverParameter solver_param;
		//	caffe::ReadSolverParamsFromTextFileOrDie(solverLocation, &solver_param);
			boost::shared_ptr<caffe::Solver<float> > solver(caffe::SolverRegistry<float>::CreateSolver(solver_param));

			const char * restore_file = solver_state_location.c_str();
			solver->Restore(restore_file);
			solver->Solve();
		}
		//TODO: Implement Examples, and make documentations
		void ODConvTrainer::fineTuning(std::string weight_file_location)
		{
		//	caffe::SolverParameter solver_param;
		//	caffe::ReadSolverParamsFromTextFileOrDie(solverLocation, &solver_param);
			// assert solver params should have a net
			const int num_train_nets = solver_param.has_net() + solver_param.has_net_param() +
				solver_param.has_train_net() + solver_param.has_train_net_param();

			const string& field_names = "net, net_param, train_net, train_net_param";
			CHECK_GE(num_train_nets, 1) << "SolverParameter must specify a train net "
				<< "using one of these fields: " << field_names;
			CHECK_LE(num_train_nets, 1) << "SolverParameter must not contain more than "
				<< "one of these fields specifying a train_net: " << field_names;

			boost::shared_ptr<caffe::Solver<float> > solver(caffe::SolverRegistry<float>::CreateSolver(solver_param));
			boost::shared_ptr<caffe::Net<float>> trained_net;

			trained_net = solver->net();
			trained_net->CopyTrainedLayersFrom(weight_file_location);
			solver->Solve();
		}	


	}
}
