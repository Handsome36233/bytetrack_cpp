#include "detector.h"
#include "bytetrack.h"
#include <chrono>
#include <iostream>


int main(int argc, char* argv[]) {
    if (argc != 3) {
        cerr << "Usage: " << argv[0] << " <onnx_model_path> <video_path>\n";
        return -1;
    }

    const char* model_path = argv[1];
    const char* video_path = argv[2];
    
    float low_conf_threshold = 0.2;
    float high_conf_threshold = 0.6;
    float iouThreshold = 0.5;
    int img_width = 640;
    int img_height = 384;
    int frame_rate = 30;

    BYTETracker tracker(frame_rate, low_conf_threshold, high_conf_threshold);
    ObjectDetector detector(img_width, img_height, low_conf_threshold, iouThreshold);
    if (!detector.init(model_path)) {
        std::cerr << "Failed to initialize model." << std::endl;
        return -1;
    }

    cv::VideoCapture cap(video_path);
    if (!cap.isOpened()) {
        std::cerr << "Error: Could not open video file.\n";
        return -1;
    }

    cv::Mat frame;

    while (cap.read(frame)) {
        std::vector<Detection> results = detector.processFrame(frame);
        std::vector<STrack> tracked_stracks = tracker.update(results);

        for (auto& track : tracked_stracks) {
            int x1 = (int)track.tlwh()[0];
            int y1 = (int)track.tlwh()[1];
            int x2 = (int)(x1 + track.tlwh()[2]);
            int y2 = (int)(y1 + track.tlwh()[3]);
            cv::rectangle(frame, cv::Rect(x1, y1, x2 - x1, y2 - y1), cv::Scalar(255, 0, 0), 2);
            cv::putText(frame, std::to_string(track.track_id), cv::Point(x1, y1 - 10), cv::FONT_HERSHEY_PLAIN, 1, cv::Scalar(0, 255, 0), 2);
        }

        cv::imshow("Detection", frame);

        if (cv::waitKey(1) == 'q') {
            break;
        }
    }

    cap.release();
    return 0;
}
