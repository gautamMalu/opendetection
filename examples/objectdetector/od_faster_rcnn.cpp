#include <detectors/global2D/detection/ODConvDetector.h>
#include <common/pipeline/ODScene.h>
#include <common/pipeline/ODDetection.h>

using std::string;

int main(int argc, char **argv){
	od::g2d::ODConvDetector *faster_rcnn = new od::g2d::ODConvDetector();
	string conf = "examples/objectdetector/faster_rcnn_models/config/voc_config.json";
	faster_rcnn->setConfig(conf);
	if (argc < 4){
		std::cout << "usage: od_faster_rcnn_cam model_definition model_weights image_source" << std::endl;
		return 0;
	}
	else{
		string model_def(argv[1]);
		string model_weight(argv[2]);
		faster_rcnn->initDetector(model_def, model_weight);
		string img_src(argv[3]);
		od::ODSceneImage *img = new od::ODSceneImage(img_src);
		od::ODDetections2D *detections = faster_rcnn->detectOmni(img);
		std::cout << "Number of detections: "<< detections->size() << std::endl;

		cv::namedWindow( "Display window",cv::WINDOW_NORMAL ); // Create a window for display.
		cv::imshow( "Display window",  detections->renderMetainfo(*img).getCVImage());// Show our image inside it.
		cv::waitKey(0); // Wait for a keystroke in the window
	}
	return 0;
}
