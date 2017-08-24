#include <detectors/global2D/detection/ODConvDetector.h>
#include <common/pipeline/ODScene.h>
#include <common/pipeline/ODDetection.h>
#include <common/utils/ODFrameGenerator.h>

using std::string;

int main(int argc, char **argv){
	od::g2d::ODConvDetector *faster_rcnn = new od::g2d::ODConvDetector();
	string conf = "examples/objectdetector/faster_rcnn_models/config/voc_config.json";
	faster_rcnn->setConfig(conf);
	string model_def(argv[1]); 
	string model_weight(argv[2]);
	faster_rcnn->initDetector(model_def, model_weight);

	//get scenes
	od::ODFrameGenerator<od::ODSceneImage, od::GENERATOR_TYPE_DEVICE> frameGenerator(0);
	//GUI
	cv::namedWindow("Overlay", cv::WINDOW_NORMAL);
	while(frameGenerator.isValid() && cv::waitKey(30) != 27)
	{
		od::ODSceneImage * scene = frameGenerator.getNextFrame();
		//Detect
		od::ODDetections2D *detections =  faster_rcnn->detectOmni(scene);
		if(detections->size() > 0)
			cv::imshow("Overlay", detections->renderMetainfo(*scene).getCVImage());
		else
			cv::imshow("Overlay", scene->getCVImage());
	}
	return 0;
}
