#include <caffe/caffe.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <algorithm>
#include <iomanip>
#include <iosfwd>
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include <boost/date_time/posix_time/posix_time.hpp>
#include <boost/filesystem/operations.hpp>
#include <boost/filesystem/path.hpp>
#include <boost/filesystem.hpp>

#include <boost/thread/thread.hpp>
#include <boost/thread/lock_factories.hpp>
#include <queue>

#include <iostream>
#include "Detector.hpp"
#include "Classifier.hpp"
#include "CvxText.h"
using namespace cv;
using namespace boost;
using namespace std;
static int ToWchar(char* &src, wchar_t* &dest, const char *locale = "zh_CN.utf8")
{
    if (src == NULL) {
        dest = NULL;
        return 0;
    }

    // 根据环境变量设置locale
    setlocale(LC_CTYPE, locale);

    // 得到转化为需要的宽字符大小
    int w_size = mbstowcs(NULL, src, 0) + 1;

    // w_size = 0 说明mbstowcs返回值为-1。即在运行过程中遇到了非法字符(很有可能使locale
    // 没有设置正确)
    if (w_size == 0) {
        dest = NULL;
        return -1;
    }

    //wcout << "w_size" << w_size << endl;
    dest = new wchar_t[w_size];
    if (!dest) {
        return -1;
    }

    int ret = mbstowcs(dest, src, strlen(src)+1);
    if (ret <= 0) {
        return -1;
    }
    return 0;
}

namespace fs = boost::filesystem;
using namespace caffe;  // NOLINT(build/namespaces)

class Detector;


void recursion(fs::path src_path, std::vector<std::string> &img_paths, int &size) {
  fs::directory_iterator end;
  for (fs::directory_iterator dir(src_path); dir != end; dir++)
  {
    std::string fn = dir->path().string();
    if(dir->path().extension() == ".jpg" || dir->path().extension() == ".JPG"
    || dir->path().extension() == ".jpeg" ) {
      img_paths.at(size++) = dir->path().string();
    }
  }
}

void GetFile(const fs::path src_path, std::string &file_path, const std::string suff) {
    fs::directory_iterator end;
    for (fs::directory_iterator dir(src_path); dir != end; dir++)
    {
        if(dir->path().filename().string().compare(suff) == 0) {
            file_path = dir->path().string();
            std::cout<<"find file: "<<file_path<<std::endl;
            return;
        }
    }
}

DEFINE_string(mean_file, "",
    "The mean file used to subtract from the input image.");
DEFINE_string(mean_value, "104,117,123",
    "If specified, can be one value or can be same as image channels"
    " - would subtract from the corresponding channel). Separated by ','."
    "Either mean_file or mean_value should be provided, not both.");
DEFINE_double(confidence_threshold, 0.01,
    "Only store detections with score higher than the threshold.");


class buffer
{
private:
  mutex mu;
  condition_variable_any cond_put;
  condition_variable_any cond_get;
  queue<Mat> stk;
  int un_read, capacity;

  bool is_full(){
      return un_read == capacity;
  }

  bool is_empty() {
      return un_read == 0;
  }

public:
  buffer(size_t n):un_read(0), capacity(n) {}
  void put(Mat frame) {
      {
          auto lock = make_unique_lock(mu);
          for(;is_full();){
              std::cout << "full waiting..." << std::endl;
              cond_put.wait(lock);
          }
          stk.push(frame);
          ++un_read;
      }
      cond_get.notify_one();
  }

  void get(Mat *img) {
      {
          auto lock = make_unique_lock(mu);
          for(;is_empty();) {
              std::cout << "empty waiting..." << std::endl;
              cond_get.wait(lock);
          }
          --un_read;
          *img = stk.front();
          stk.pop();
      }
      cond_put.notify_one();
  }
};


buffer video_buf(5);
buffer show_buf(5);

void video_producer(string path) {
    cv::VideoCapture cap = cv::VideoCapture(path);
    cv::Mat frame;
    while (cap.read(frame)) {
//        std::cout << "put " << i << std::endl;
        video_buf.put(frame);
    }
//    buf.put(NULL);
}
void show_consumer() {
    while(1) {

        cv::Mat img;
        show_buf.get(&img);
        cv::imshow("demo", img);
        cv::waitKey(1);
    }
//   std::cout << "get" << x << std::endl;
}




int main(int argc, char** argv) {
    ::google::InitGoogleLogging(argv[0]);
    FLAGS_alsologtostderr = 1;
#ifndef GFLAGS_GFLAGS_H_
    namespace gflags = google;
#endif
    gflags::SetUsageMessage("Do detection using SSD mode.\n"
          "Usage:\n"
          "    ssd_detect [FLAGS] model_file weights_file list_file\n");
    gflags::ParseCommandLineFlags(&argc, &argv, true);


    const string& testPath = "./test/imgs/";
    const string& model_path = "./model";

    string file_name;
    GetFile(model_path, file_name, "model.caffemodel");
    const string& model_file = "./model/deploy.prototxt";
    GetFile(model_path, file_name, "deploy.prototxt");
    const string& weights_file = "./model/model.caffemodel";
    const string& mean_file = "./model/labelmap_voc.prototxt";
    const string& mean_value = FLAGS_mean_value;
    const float confidence_threshold = FLAGS_confidence_threshold;
    // Initialize the network.
    Detector people_detector("./model/deploy_people.prototxt", "./model/model_people.caffemodel", "./model/labelmap_people.prototxt", mean_value);
    Classifier classifier("./model/deploy_people_attribe.prototxt", "./model/model_people_attribe.caffemodel",
        "./model/mean_people_attribe.binaryproto", "./model/label_people_attribe.txt");
    //test video

    int sub = 0;
    cv::VideoCapture cap = cv::VideoCapture("/home/rui/demo.mp4");
    thread_group tg;
    tg.create_thread(bind(video_producer,"/home/rui/demo.mp4"));
    tg.create_thread(bind(show_consumer));

    float fps = cap.get(CV_CAP_PROP_FPS);

    cv::Mat img;

    int frame_width = cap.get(CV_CAP_PROP_FRAME_WIDTH);
    int frame_height = cap.get(CV_CAP_PROP_FRAME_HEIGHT);




    CvxText text("./demo.ttf"); //指定字体
    cv::Scalar size1{ 60, 0.5, 0.1, 0 }; // (字体大小, 无效的, 字符间距, 无效的 }

    text.setFont(nullptr, &size1, nullptr, 0);

    sub = 0;
    int step = 200;
    while (1){
        video_buf.get(&img);
        cv::Mat showimg = img.clone();
        float scalar = img.cols/float(1366);
        cv::Size size_show = cv::Size(int(img.cols/scalar), int(img.rows/scalar));
        std::vector<cv::Rect> peoples_rect;
        people_detector.DetectReturnImg(img, showimg, peoples_rect);
        for(int people_id = 0; people_id < peoples_rect.size(); ++people_id){
            cv::Rect peo_rect = peoples_rect[people_id];
            cv::Mat peo_img = img(peo_rect).clone();

            std::vector<Prediction> prediction = classifier.Classify(peo_img);
            std::cout << prediction[0].first<< ": " << prediction[0].second << std::endl;
            std::string show_text = prediction[0].first; //+ ": " + std::to_string(prediction[0].second);
            if(prediction[0].second < 0.3)
                show_text = "";
            const char* str = show_text.c_str();
            wchar_t *w_str;
            char* show_str = const_cast<char *>(str);
            ToWchar(show_str,w_str);
            text.putText(showimg, w_str,  cv::Point(peo_rect.x,peo_rect.y), cv::Scalar(252, 245, 202));

        }

        cv::resize(showimg, showimg, size_show);
        show_buf.put(showimg);
        for(int n = 0; n < sub ; n++) {
            cap.read(img);
        }
    }
  return 0;
}
