 /// adding funtions related to average precision

#include <sys/time.h>
#include <boost/preprocessor.hpp>
#include <boost/filesystem.hpp>
#include <boost/algorithm/string.hpp>
#include <boost/assert.hpp> // for assert message
#include <opencv2/core/core.hpp>

#include <fstream>
#include <sstream>
#include <iostream>
#include <glob.h>
#include<numeric>
#include<algorithm>
#include<vector>

#include "common/pipeline/ODDetection.h"
#include "common/utils/utils.h"
using namespace std;
using namespace od;
using namespace cv;

float iou(cv::Rect gt, cv::Rect res){
    cv::Rect intersection_ = gt & res;
    cv::Rect union_ = gt | res;
    return float(intersection_.area())/float(union_.area());
}

float ap(std::vector<od::ODDetections2D *> gt, std::vector<od::ODDetections2D *> res, const std::string cls, float thresh)
{
    BOOST_ASSERT_MSG(gt.size() == res.size(), "Number of images should be same for predictions and groud truths");

    std::vector<ODDetections2D*> gtc;// ground truth detections for given class
    int npos=0; // number of GT bbox for the given class

    std::vector<cv::Rect> BB; // bounding boxes for results for the given class
    std::vector<int> image_ids; //image index for the BBs 
    std::vector<double> confidence; //confidence for the BBs
    for(int i =0; i<gt.size(); i++){
        ODDetections2D *curr = new ODDetections2D();
        for(int j=0;j<gt[i]->size();j++){
            if (gt[i]->at(j)->getId() == cls){
                curr->push_back(gt[i]->at(j));
                npos++;
            }            
        }
        gtc.push_back(curr);
        // filter the result BBoxes also
        for (int k=0;k<res[i]->size(); k++){
            if (res[i]->at(k)->getId() == cls){
                image_ids.push_back(i);
                BB.push_back(res[i]->at(k)->getBoundingBox());
                confidence.push_back(res[i]->at(k)->getConfidence());
            }
        }
    }

    //sort the result BBoxes based on Confidence
    std::vector<int> sorted_index(confidence.size());
    std::iota(sorted_index.begin(), sorted_index.end(), 0);
    std::sort(std::begin(sorted_index), std::end(sorted_index),
            [&](int i1, int i2) { return confidence[i1] > confidence[i2]; } );

    std::vector<cv::Rect> sort_BB(BB.size()); 
    std::vector<int> sort_image_ids(image_ids.size()); 
    std::vector<double> sort_confidence(confidence.size());

    for(int i=0;i<image_ids.size();i++){
        sort_image_ids[i] = image_ids[sorted_index[i]];
        sort_BB[i] = BB[sorted_index[i]];
        sort_confidence[i] = confidence[sorted_index[i]];
    }
    BB.clear();
    image_ids.clear();
    confidence.clear();


    int nd = sort_image_ids.size();
    /*flags for true positives and false positive 
      for all predicted bboxes*/
    std::vector<int>tp(nd,0);
    std::vector<int>fp(nd,0);
    int image_index;
    cv::Rect curr_res;
    float max_overlap,curr_overlap;
    int max_overlap_index;


    for(int i=0; i<nd; i++){
        image_index = sort_image_ids[i];
        curr_res = sort_BB[i];

        if (gtc[image_index]->size() > 0){
            max_overlap = -1.0;
            max_overlap_index = -1;
            for(int j=0; j< gtc[image_index]->size(); j++){
                curr_overlap = iou(gtc[image_index]->at(j)->getBoundingBox(), curr_res);
                if (curr_overlap > max_overlap){
                    max_overlap = curr_overlap;
                    max_overlap_index = j;
                }
            }
            //Check if max_overlap is greater than thresh
            if (max_overlap > thresh){
                tp[i] = 1;
                // Remove the detected GT 
                gtc[image_index]->remove(max_overlap_index);
            }
            else // false positive
                fp[i] = 1;
        }
    }
    //Compute Precision and Recall
    // cummulative sum
    std::partial_sum(fp.begin(), fp.end(), fp.begin());
    std::partial_sum(tp.begin(), tp.end(), tp.begin());

    std::vector<float> rec(nd);
    std::vector<float> pre(nd);
    float eps = 0.00001; 
    // avoid divide by zero 
    for(int i=0; i<nd; i++){
        rec[i] = float(tp[i])/float(npos);
        pre[i] = float(tp[i])/(std::max(eps, float(tp[i] + fp[i])));
    }
    /* correct AP calculation
       first append sentinel values at the end*/
    rec.insert(rec.begin(), 0.0);
    rec.push_back(1.0);
    pre.insert(pre.begin(), 0.0);
    pre.push_back(0.0);
    //compute precision envelope
    for(int i=pre.size() -1; i>0; i--)
        pre[i-1] = std::max(pre[i-1], pre[i]);

    /* to calculate area under PR curve, look for points
       where X axis (recall) changes value */
    float ap = 0;
    // if recall changes value then rec[i+1] - rec[i] > 0
    for(int i=0; i< pre.size()-1; i++)
        ap = ap + ((rec[i+1] - rec[i])*pre[i+1]); 

    return ap;
}

