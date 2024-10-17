#ifndef DETECTOR_H
#define DETECTOR_H

#include <opencv2/opencv.hpp>
#include <onnxruntime_cxx_api.h>
#include <iostream>
#include <vector>
#include <memory>

using namespace std;

struct Detection {
    float left;
    float top; 
    float width;
    float height;
    int class_id;
    float score;

    Detection(int l, int t, int w, int h, int cid, float s) 
        : left(l), top(t), width(w), height(h), class_id(cid), score(s) {}
};

class ObjectDetector {
public:
    ObjectDetector();
    ObjectDetector(int img_width, int img_height, float confidence_thres, float iou_thres);
    
    bool init(const string& model_path);
    vector<Detection> processFrame(cv::Mat& frame);

private:
    unique_ptr<Ort::Session> session;
    Ort::SessionOptions session_options;
    Ort::AllocatorWithDefaultOptions allocator;
    vector<const char*> inputNodeNames;
    vector<const char*> outputNodeNames;
    int img_width, img_height;
    float confidence_threshold, iou_threshold;

    void PrintSessionInfo(const Ort::Session& session);
    void InitializeInputOutputNames();
    void BlobFromImage(cv::Mat& img, float* blob);
    vector<Detection> extractResults(const vector<Ort::Value>& outputTensor, float resizeScalesW, float resizeScalesH, cv::Mat& show_img);
};

#endif
