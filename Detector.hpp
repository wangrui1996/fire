#ifndef MAIN_DETECTOR_HPP
#define MAIN_DETECTOR_HPP

#include <caffe/caffe.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <stdlib.h>
using namespace caffe;

class Detector {
 public:
  Detector(const string& model_file,
           const string& weights_file,
           const string& label_map_file,
           const string& mean_value);

  std::vector<vector<float> > Detect(const cv::Mat& img);

  void DetectReturnImg(cv::Mat img, cv::Mat &imgshow, std::vector<cv::Rect> &rects, float threshold=0.2);

 private:
  void SetMean(const string& mean_file, const string& mean_value);

  void WrapInputLayer(std::vector<cv::Mat>* input_channels);

  void Preprocess(const cv::Mat& img,
                  std::vector<cv::Mat>* input_channels);



 private:
  shared_ptr<Net<float> > net_;
  cv::Size input_geometry_;
  int num_channels_;
  cv::Mat mean_;
  map<int, string> label_to_display_name_;
  vector<cv::Scalar> color;
  int index_;
};



#endif //MAIN_DETECTOR_HPP
