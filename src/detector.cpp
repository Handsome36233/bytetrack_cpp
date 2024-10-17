#include "detector.h"
#include <chrono>

ObjectDetector::ObjectDetector()
    : img_width(640), img_height(384), 
      confidence_threshold(0.3f), iou_threshold(0.5f) {}

ObjectDetector::ObjectDetector(int img_width, int img_height, float confidence_thres, float iou_thres)
    : img_width(img_width), img_height(img_height), 
      confidence_threshold(confidence_thres), iou_threshold(iou_thres) {}

bool ObjectDetector::init(const string& model_path) {
    try {
        Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "yolov8_inference");
        session_options.SetIntraOpNumThreads(1);
        session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);
        session = make_unique<Ort::Session>(env, model_path.c_str(), session_options);
        allocator = Ort::AllocatorWithDefaultOptions();
        PrintSessionInfo(*session);
        InitializeInputOutputNames();
        return true;
    } catch (const std::exception& e) {
        cerr << "Error initializing model: " << e.what() << endl;
        return false;
    }
}

vector<Detection> ObjectDetector::processFrame(cv::Mat& frame) {
    cv::Mat resized_frame;
    cv::Mat input_frame = frame.clone();
    cv::cvtColor(input_frame, resized_frame, cv::COLOR_BGR2RGB);
    float resizeScalesW = resized_frame.cols / float(img_width);
    float resizeScalesH = resized_frame.rows / float(img_height);
    cv::resize(resized_frame, resized_frame, cv::Size(img_width, img_height));

    // 转tensor
    float* blob = new float[resized_frame.total() * 3];
    BlobFromImage(resized_frame, blob);
    vector<int64_t> inputNodeDims = { 1, 3, img_height, img_width };
    Ort::Value inputTensor = Ort::Value::CreateTensor<float>(
        Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU), blob, 3 * img_width * img_height,
        inputNodeDims.data(), inputNodeDims.size());

    // 模型推理
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "yolov8_inference");
    auto start_time = std::chrono::high_resolution_clock::now();
    auto outputTensor = session->Run(Ort::RunOptions{ nullptr }, inputNodeNames.data(), &inputTensor, 1, outputNodeNames.data(),
        outputNodeNames.size());
    auto end_time = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> total_duration = end_time - start_time;

    return extractResults(outputTensor, resizeScalesW, resizeScalesH, frame);
}

void ObjectDetector::PrintSessionInfo(const Ort::Session& session) {
    auto input_count = session.GetInputCount();
    auto output_count = session.GetOutputCount();
    cout << "Input Count: " << input_count << endl;
    cout << "Output Count: " << output_count << endl;

    for (int i = 0; i < input_count; ++i) {
        string input_name = session.GetInputNameAllocated(i, allocator).get();
        vector<int64_t> input_shape = session.GetInputTypeInfo(i).GetTensorTypeAndShapeInfo().GetShape();
        cout << "Input " << i << " Name: " << input_name << endl;
        cout << "Input " << i << " Shape: ";
        for (const auto& dim : input_shape) cout << dim << ' ';
        cout << endl;
    }

    for (int i = 0; i < output_count; ++i) {
        string output_name = session.GetOutputNameAllocated(i, allocator).get();
        vector<int64_t> output_shape = session.GetOutputTypeInfo(i).GetTensorTypeAndShapeInfo().GetShape();
        cout << "Output " << i << " Name: " << output_name << endl;
        cout << "Output " << i << " Shape: ";
        for (const auto& dim : output_shape) cout << dim << ' ';
        cout << endl;
    }
}

void ObjectDetector::InitializeInputOutputNames() {
    size_t inputNodesNum = session->GetInputCount();
    for (size_t i = 0; i < inputNodesNum; i++) {
        Ort::AllocatedStringPtr input_node_name = session->GetInputNameAllocated(i, allocator);
        char* temp_buf = new char[50];
        strcpy(temp_buf, input_node_name.get());
        inputNodeNames.push_back(temp_buf);
    }
    size_t OutputNodesNum = session->GetOutputCount();
    for (size_t i = 0; i < OutputNodesNum; i++) {
        Ort::AllocatedStringPtr output_node_name = session->GetOutputNameAllocated(i, allocator);
        char* temp_buf = new char[10];
        strcpy(temp_buf, output_node_name.get());
        outputNodeNames.push_back(temp_buf);
    }
}

void ObjectDetector::BlobFromImage(cv::Mat& img, float* blob) {
    int channels = img.channels();
    int imgHeight = img.rows;
    int imgWidth = img.cols;

    for (int c = 0; c < channels; c++) {
        for (int h = 0; h < imgHeight; h++) {
            for (int w = 0; w < imgWidth; w++) {
                blob[c * imgWidth * imgHeight + h * imgWidth + w] = static_cast<float>(img.at<cv::Vec3b>(h, w)[c]) / 255.0f;
            }
        }
    }
}

vector<Detection> ObjectDetector::extractResults(const vector<Ort::Value>& outputTensor, float resizeScalesW, float resizeScalesH, cv::Mat& show_img) {
    Ort::TypeInfo typeInfo = outputTensor.front().GetTypeInfo();
    auto tensor_info = typeInfo.GetTensorTypeAndShapeInfo();
    vector<int64_t> outputNodeDims = tensor_info.GetShape();
    auto output = outputTensor.front().GetTensorData<float>();
    int signalResultNum = outputNodeDims[1];
    int strideNum = outputNodeDims[2];

    vector<int> class_ids;
    vector<float> confidences;
    vector<cv::Rect> boxes;

    for (int i = 0; i < strideNum; ++i) {
        float tmp_class_score[signalResultNum - 4];
        for (int j = 0; j < signalResultNum - 4; ++j)
            tmp_class_score[j] = output[strideNum * (j + 4) + i];
        auto classesScores = max_element(tmp_class_score, tmp_class_score + signalResultNum - 4);
        auto classId = distance(tmp_class_score, classesScores);

        if (*classesScores > confidence_threshold) {
            confidences.push_back(*classesScores);
            class_ids.push_back(classId);
            float x = output[strideNum * 0 + i];
            float y = output[strideNum * 1 + i];
            float w = output[strideNum * 2 + i];
            float h = output[strideNum * 3 + i];

            int left = int((x - 0.5 * w) * resizeScalesW);
            int top = int((y - 0.5 * h) * resizeScalesH);
            int width = int(w * resizeScalesW);
            int height = int(h * resizeScalesH);

            boxes.push_back(cv::Rect(left, top, width, height));
        }
    }

    vector<int> nmsResult;
    cv::dnn::NMSBoxes(boxes, confidences, confidence_threshold, iou_threshold, nmsResult);
    vector<Detection> oResult;

    for (int i = 0; i < nmsResult.size(); ++i) {
        int idx = nmsResult[i];
        Detection result(boxes[idx].x, boxes[idx].y, boxes[idx].width, boxes[idx].height, class_ids[idx], confidences[idx]);
        oResult.push_back(result);
    }

    // for (auto& re : oResult) {
    //     cout<< "left: " << re.left << " top: " << re.top << " width: " << re.width << " height: " << re.height << " class_id: " << re.class_id << " score: " << re.score << endl;
    //     cv::rectangle(show_img, cv::Rect(re.left, re.top, re.width, re.height), cv::Scalar(0, 255, 0), 3);
    // }

    return oResult;
}

