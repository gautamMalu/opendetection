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
			#ifdef WITH_GPU
			solver_param.set_solver_mode(caffe::SolverParameter_SolverMode_GPU); //GPU
			#else
			solver_param.set_solver_mode(caffe::SolverParameter_SolverMode_CPU); //CPU
			#endif
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

		void ODConvTrainer::test(const std::string model_def, const std::string model_weights, int test_iter){
			//Check test net definition availibity in solver
			/*const string& field_names = "net, net_param, test_net, test_net_param";

			std::string model_def_;
			if (sovler_param.has_net())
				model_def_ = solver_param.net();
			else if (sovler_param.has_net_param())
				model_def_ = solver_param.net_param();
			else if (sovler_param.has_test_net())
				model_def_ = solver_param.test_net();
			else if (sovler_param.has_test_net_param())
				model_def_ = solver_param.net();
			else
				std::cout << "SolverParameter must specify a test net "
					<< "using one of these fields: " << field_names;
			*/
			caffe::Net<float> net_(model_def, caffe::TEST);
	//		caffe::shared_ptr<caffe::Net<float> > net_;
			#ifdef WITH_GPU
			Caffe::set_mode(Caffe::GPU);
			#else
			Caffe::set_mode(Caffe::CPU);
			#endif

			net_.CopyTrainedLayersFrom(model_weights);
			std::cout << "Running for " << test_iter << " iterations."<<std::endl;

			vector<int> test_score_output_id;
			vector<float> test_score;
			float loss = 0;
			for (int i = 0; i < test_iter; ++i) {
				float iter_loss;
				const vector<Blob<float>*>& result = net_.Forward(&iter_loss);
				loss += iter_loss;
				int idx = 0;
				for (int j = 0; j < result.size(); ++j) {
					const float* result_vec = result[j]->cpu_data();
					for (int k = 0; k < result[j]->count(); ++k, ++idx) {
						const float score = result_vec[k];
						if (i == 0) {
							test_score.push_back(score);
							test_score_output_id.push_back(j);
						} 
						else {
							test_score[idx] += score;
						}
						const std::string& output_name = net_.blob_names()[
							net_.output_blob_indices()[j]];
						std::cout << "Batch " << i << ", " << output_name << " = " << score << std::endl;
					}
				}
			}
			loss /= test_iter;
			std::cout << "Loss: " << loss << std::endl;
			for (int i = 0; i < test_score.size(); ++i) {
				const std::string& output_name = net_.blob_names()[
					net_.output_blob_indices()[test_score_output_id[i]]];
				const float loss_weight = net_.blob_loss_weights()[
					net_.output_blob_indices()[test_score_output_id[i]]];
				std::ostringstream loss_msg_stream;
				const float mean_score = test_score[i] / test_iter;
				if (loss_weight) {
					loss_msg_stream << " (* " << loss_weight
						<< " = " << loss_weight * mean_score << " loss)";
				}
				std::cout << output_name << " = " << mean_score << loss_msg_stream.str() << std::endl;
			}
		}

	}

}
