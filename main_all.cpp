// main_all.cpp
#include <iostream>
#include <iomanip>
#include <filesystem>
#include <fstream>
#include <vector>
#include <chrono>
#include <memory>
#include <string>
#include <map>
#include <sstream>

#include <opencv2/opencv.hpp>

// ========== 1. 外部头文件 ==========
#include "BYTETracker_obb.h"
#include "yolo_obb.h"
// ==================================

namespace fs = std::filesystem;
using std::chrono::steady_clock;
using std::chrono::microseconds;
using std::chrono::duration_cast;

/* ----------------------------------------------------------
 * 通用
 * ---------------------------------------------------------- */
template<typename... Args>
static std::string str_format(const std::string& fmt, Args... args)
{
    int sz = std::snprintf(nullptr, 0, fmt.c_str(), args...) + 1;
    std::unique_ptr<char[]> buf(new (std::nothrow) char[sz]);
    if (!buf) return "";
    std::snprintf(buf.get(), sz, fmt.c_str(), args...);
    return std::string(buf.get(), buf.get() + sz - 1);
}

/* ----------------------------------------------------------
 * 配置解析结构
 * ---------------------------------------------------------- */
struct Config
{
    std::string task;               // "detect" 或 "track"
    std::string modelPath;
    std::string imageIn;            // 单张图或图目录
    std::string imgOut;             // 检测/跟踪结果图保存路径（文件或目录）
    std::string labelOut;           // 检测标签保存路径（文件或目录）
    std::string trackImgOut;        // 仅跟踪任务时有效，可空
    int         classNum      = 5;
    float       confThresh    = 0.25f;
    float       nmsThresh     = 0.45f;
    int         modelW        = 640;
    int         modelH        = 640;
    int         modelBoxNum   = 8400;
    std::string modelType     = "YOLO11_OBB";  // YOLO11_OBB 或 YOLOV8_OBB
    std::string classesFile   = "../classes.txt";
    std::vector<std::string> classLabels;

    // 解析配置文件
    bool loadFromFile(const std::string& configPath)
    {
        std::ifstream file(configPath);
        if (!file.is_open()) {
            std::cerr << "Failed to open config file: " << configPath << std::endl;
            return false;
        }

        std::string line;
        while (std::getline(file, line)) {
            // 跳过注释和空行
            if (line.empty() || line[0] == '#') continue;
            
            size_t pos = line.find('=');
            if (pos == std::string::npos) continue;
            
            std::string key = line.substr(0, pos);
            std::string value = line.substr(pos + 1);
            
            // 去除首尾空格
            key.erase(0, key.find_first_not_of(" \t"));
            key.erase(key.find_last_not_of(" \t") + 1);
            value.erase(0, value.find_first_not_of(" \t"));
            value.erase(value.find_last_not_of(" \t") + 1);
            
            if (key == "task") task = value;
            else if (key == "model_path") modelPath = value;
            else if (key == "image_input") imageIn = value;
            else if (key == "image_output") imgOut = value;
            else if (key == "label_output") labelOut = value;
            else if (key == "track_image_output") trackImgOut = value;
            else if (key == "class_num") classNum = std::stoi(value);
            else if (key == "conf_thresh") confThresh = std::stof(value);
            else if (key == "nms_thresh") nmsThresh = std::stof(value);
            else if (key == "model_width") modelW = std::stoi(value);
            else if (key == "model_height") modelH = std::stoi(value);
            else if (key == "model_box_num") modelBoxNum = std::stoi(value);
            else if (key == "model_type") modelType = value;
            else if (key == "classes_file") classesFile = value;
        }

        // 加载类别标签
        if (!loadClassLabels()) {
            std::cerr << "Failed to load class labels from: " << classesFile << std::endl;
            return false;
        }

        return true;
    }

private:
    bool loadClassLabels()
    {
        std::ifstream file(classesFile);
        if (!file.is_open()) {
            return false;
        }

        classLabels.clear();
        std::string line;
        while (std::getline(file, line)) {
            // 去除首尾空格
            line.erase(0, line.find_first_not_of(" \t\r\n"));
            line.erase(line.find_last_not_of(" \t\r\n") + 1);
            if (!line.empty()) {
                classLabels.push_back(line);
            }
        }

        return !classLabels.empty();
    }
};

/* ----------------------------------------------------------
 * YoloOBB 封装
 * ---------------------------------------------------------- */
class YoloOBBWrapper
{
public:
    YoloOBBWrapper(const InferenceConfig& cfg, const std::string& modelType, 
                   const std::vector<std::string>& labels) 
        : cfg_(cfg), modelType_(modelType), labels_(labels), inference_(cfg_) {}
    
    bool Init() { return inference_.initialize(); }

    // 纯检测接口：返回 OBB 和 Object_OBB
    bool Detect(const std::string& imagePath,
                std::vector<OBBBoundingBox>& outOBBs,
                std::vector<Object_OBB>& outObjects)
    {
        if (!inference_.preprocessImage(imagePath)) return false;
        std::vector<InferenceOutput> outs;
        if (!inference_.runModelInference(outs)) return false;

        cv::Mat src = cv::imread(imagePath);
        if (src.empty()) return false;

        float* ptr = static_cast<float*>(outs[0].data.get());
        OBBPostProcessor post(cfg_.modelOutputBoxNum, cfg_.classNum);
        
        // 根据模型类型选择不同的后处理方法
        std::vector<OBBBoundingBox> boxes;
        if (modelType_ == "YOLO11_OBB") {
            boxes = post.parseOutput_YOLO11OBB(ptr, 
                                     src.cols, src.rows,
                                     cfg_.modelWidth, cfg_.modelHeight,
                                     cfg_.confidenceThreshold);
        } else if (modelType_ == "YOLOV8_OBB") {
            boxes = post.parseOutput(ptr, 
                                   src.cols, src.rows,
                                   cfg_.modelWidth, cfg_.modelHeight,
                                   cfg_.confidenceThreshold);
        } else {
            std::cerr << "Unsupported model type: " << modelType_ << std::endl;
            return false;
        }
        
        outOBBs = OBBNMSProcessor::applyNMS(boxes, cfg_.nmsThreshold);

        // 转换为 Object_OBB 格式
        outObjects.clear();
        for (const auto& obb : outOBBs)
        {
            Object_OBB obj;
            obj.label = static_cast<int>(obb.classIndex);
            obj.prob  = obb.confidence;
            obj.rect.center.x = obb.cx;
            obj.rect.center.y = obb.cy;
            obj.rect.size.width = obb.width;
            obj.rect.size.height = obb.height;
            obj.rect.angle = obb.angle;
            outObjects.push_back(obj);
        }
        return true;
    }

    const std::vector<std::string>& getLabels() const { return labels_; }

private:
    InferenceConfig cfg_;
    std::string modelType_;
    std::vector<std::string> labels_;
    YOLOOBBInference inference_;
};

// YoloOBBWrapper类的Detect方法修改

/* ----------------------------------------------------------
 * 绘制 OBB
 * ---------------------------------------------------------- */
static void drawOBB(cv::Mat& img, const OBBBoundingBox& obb,
                    const cv::Scalar& color, const std::string& text = "")
{
    auto pts_f = obb.getCornerPoints();
    std::vector<cv::Point> pts;
    for (auto& p : pts_f) pts.emplace_back(cv::Point(int(std::round(p.x)), int(std::round(p.y))));
    const cv::Point* ppt = pts.data();
    int n = int(pts.size());
    cv::polylines(img, &ppt, &n, 1, true, color, 2);
    cv::circle(img, cv::Point(int(std::round(obb.cx)), int(std::round(obb.cy))),
               3, color, -1);
    std::string txt = text.empty()
        ? cv::format("cls:%d %.2f", int(obb.classIndex), obb.confidence)
        : text;
    int baseline = 0;
    cv::Size ts = cv::getTextSize(txt, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseline);
    int tx0 = std::max(0, int(std::round(obb.cx - ts.width / 2.f)));
    int ty0 = std::max(0, int(std::round(obb.cy - obb.height / 2 - 6)));
    cv::putText(img, txt, cv::Point(tx0, ty0), cv::FONT_HERSHEY_SIMPLEX,
                0.5, color, 1);
}

// 重载版本：从 cv::RotatedRect 绘制 OBB
static void drawOBBFromRotatedRect(cv::Mat& img, const cv::RotatedRect& rect,
                                   const cv::Scalar& color, const std::string& text = "")
{
    cv::Point2f vertices[4];
    rect.points(vertices);
    
    std::vector<cv::Point> pts;
    for (int i = 0; i < 4; i++) {
        pts.emplace_back(cv::Point(int(std::round(vertices[i].x)), int(std::round(vertices[i].y))));
    }
    
    const cv::Point* ppt = pts.data();
    int n = int(pts.size());
    cv::polylines(img, &ppt, &n, 1, true, color, 2);
    cv::circle(img, cv::Point(int(std::round(rect.center.x)), int(std::round(rect.center.y))),
               3, color, -1);
    
    if (!text.empty()) {
        int baseline = 0;
        cv::Size ts = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseline);
        int tx0 = std::max(0, int(std::round(rect.center.x - ts.width / 2.f)));
        int ty0 = std::max(0, int(std::round(rect.center.y - rect.size.height / 2 - 6)));
        cv::putText(img, text, cv::Point(tx0, ty0), cv::FONT_HERSHEY_SIMPLEX,
                    0.5, color, 1);
    }
}

static void drawOBBFromRotatedRect(cv::Mat& img, const OBBBoundingBox& rect,
                                   const cv::Scalar& color, const std::string& text = "")
{
    std::vector<cv::Point2f> vertices = rect.getCornerPoints();
    
    std::vector<cv::Point> pts;
    for (int i = 0; i < 4; i++) {
        pts.emplace_back(cv::Point(int(std::round(vertices[i].x)), int(std::round(vertices[i].y))));
    }
    
    const cv::Point* ppt = pts.data();
    int n = int(pts.size());
    cv::polylines(img, &ppt, &n, 1, true, color, 2);
    cv::circle(img, cv::Point(int(std::round(rect.cx)), int(std::round(rect.cy))),
               3, color, -1);
    
    if (!text.empty()) {
        int baseline = 0;
        cv::Size ts = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseline);
        int tx0 = std::max(0, int(std::round(rect.cx - ts.width / 2.f)));
        int ty0 = std::max(0, int(std::round(rect.cy - rect.height / 2 - 6)));
        cv::putText(img, text, cv::Point(tx0, ty0), cv::FONT_HERSHEY_SIMPLEX,
                    0.5, color, 1);
    }
}

static void printUsage(const char* prog)
{
    std::cout <<
    "\nUsage:\n"
    "  " << prog << " <config_file>\n\n"
    "  Example:\n"
    "    " << prog << " config.txt\n\n"
    "  配置文件格式说明：\n"
    "    task=detect                    # 任务类型: detect 或 track\n"
    "    model_path=model.om            # 模型路径\n"
    "    image_input=/path/to/input     # 输入图片或目录\n"
    "    image_output=/path/to/output   # 输出图片目录\n"
    "    label_output=/path/to/labels   # 输出标签目录\n"
    "    model_type=YOLO11_OBB          # 模型类型: YOLO11_OBB 或 YOLOV8_OBB\n"
    "    classes_file=classes.txt       # 类别标签文件\n";
}

/* ----------------------------------------------------------
 * 纯检测任务（单张图 或 连续编号文件夹）
 * ---------------------------------------------------------- */
static void runDetect(const Config& config)
{
    InferenceConfig cfg;
    cfg.modelPath      = config.modelPath;
    cfg.modelWidth     = config.modelW;
    cfg.modelHeight    = config.modelH;
    cfg.modelOutputBoxNum = config.modelBoxNum;
    cfg.classNum       = config.classNum;
    cfg.confidenceThreshold = config.confThresh;
    cfg.nmsThreshold   = config.nmsThresh;

    YoloOBBWrapper detector(cfg, config.modelType, config.classLabels);
    if (!detector.Init()) { std::cerr << "Init detector failed.\n"; return; }

    /* 如果是单张图，列表只有 1 项；如果是目录，按 6 位编号扫描 */
    std::vector<fs::path> inFiles;
    if (fs::is_regular_file(config.imageIn))
        inFiles.emplace_back(config.imageIn);
    else
    {
        for (int no = 1; ; ++no)
        {
            fs::path p = fs::path(config.imageIn) / str_format("%06d.jpg", no);
            if (!fs::exists(p)) break;
            inFiles.push_back(p);
        }
    }
    if (inFiles.empty()) { std::cerr << "No input images.\n"; return; }

    fs::create_directories(config.imgOut);
    fs::create_directories(config.labelOut);

    const cv::Scalar colors[] = {
        {255,0,0},{0,255,0},{0,0,255},{255,255,0},{255,0,255},{0,255,255}
    };

    int64_t totalUs = 0;
    int     done    = 0;

    for (const auto& inPath : inFiles)
    {
        cv::Mat img = cv::imread(inPath.string());
        if (img.empty()) continue;

        std::vector<OBBBoundingBox> obbs;
        std::vector<Object_OBB> objects;
        auto t0 = steady_clock::now();
        bool ok = detector.Detect(inPath.string(), obbs, objects);
        auto t1 = steady_clock::now();
        if (!ok) continue;

        int64_t us = duration_cast<microseconds>(t1 - t0).count();
        totalUs += us;
        ++done;

        /* 画框 */
        cv::Mat res = img.clone();
        const auto& labels = detector.getLabels();
        for (const auto& o : obbs) {
            std::string label = (o.classIndex < labels.size()) ? 
                               labels[o.classIndex] : "unknown";
            std::string text = label + ":" + std::to_string(o.confidence).substr(0, 4);
            drawOBB(res, o, colors[o.classIndex % 6], text);
        }

        /* 输出文件：与输入同名 */
        fs::path outImg   = fs::path(config.imgOut)   / (inPath.stem().string() + ".jpg");
        fs::path outLabel = fs::path(config.labelOut) / (inPath.stem().string() + ".txt");

        cv::imwrite(outImg.string(), res);

        std::ofstream fo(outLabel.string());
        for (const auto& o : obbs)
        {
            auto pts = o.getCornerPoints();
            fo << o.classIndex << ' ' << o.confidence;
            for (const auto& p : pts) fo << ' ' << p.x << ' ' << p.y;
            fo << '\n';
        }
    }

    if (done == 0) { std::cerr << "No frame processed.\n"; return; }

    double fps = done * 1'000'000.0 / totalUs;
    std::cout << "[Detect] 平均 FPS = " << fps
              << "  (共 " << done << " 张，总耗时 "
              << totalUs / 1000 << " ms)\n";
}

/* ----------------------------------------------------------
 * 检测+跟踪任务（连续编号图像序列）
 * ---------------------------------------------------------- */
static void runTrack(const Config& config)
{
    InferenceConfig cfg;
    cfg.modelPath = config.modelPath;
    cfg.modelWidth = config.modelW; cfg.modelHeight = config.modelH;
    cfg.modelOutputBoxNum = config.modelBoxNum;
    cfg.classNum = config.classNum;
    cfg.confidenceThreshold = config.confThresh;
    cfg.nmsThreshold = config.nmsThresh;

    YoloOBBWrapper detector(cfg, config.modelType, config.classLabels);
    if (!detector.Init()) { std::cerr << "Init detector failed.\n"; return; }

    BYTETracker_obb tracker(30, 30);

    fs::create_directories(config.imgOut);
    fs::create_directories(config.labelOut);
    if (!config.trackImgOut.empty()) fs::create_directories(config.trackImgOut);

    int64_t totalDetUs = 0, totalTrkUs = 0;
    int frameCnt = 0;

    for (int no = 1; ; ++no)
    {
        std::string imgPath = str_format("%s/%06d.jpg", config.imageIn.c_str(), no);
        if (!fs::exists(imgPath)) break;
        ++frameCnt;

        cv::Mat img = cv::imread(imgPath);
        if (img.empty()) continue;

        /* ---- detect ---- */
        std::vector<OBBBoundingBox> obbs; 
        std::vector<Object_OBB> objects;
        auto t0 = steady_clock::now();
        bool ok = detector.Detect(imgPath, obbs, objects);
        auto t1 = steady_clock::now();
        if (!ok) continue;
        int64_t detUs = duration_cast<microseconds>(t1 - t0).count();
        totalDetUs += detUs;

        /* ---- track ---- */
        t0 = steady_clock::now();
        std::vector<STrack_obb> stracks = tracker.update(objects);
        t1 = steady_clock::now();
        int64_t trkUs = duration_cast<microseconds>(t1 - t0).count();
        totalTrkUs += trkUs;

        /* 关联 obb -> track_id，使用ProbIoU计算 */
        std::map<int, OBBBoundingBox> tid2obb;

        for (const auto& trk : stracks)
        {
            float bestIou = 0; 
            int bestIdx = -1;
            
            // 构造跟踪目标的 OBBBoundingBox
            OBBBoundingBox trkOBB;
            trkOBB.cx = trk.tlwh[0];      // cx
            trkOBB.cy = trk.tlwh[1];      // cy
            trkOBB.width = trk.tlwh[2];   // width
            trkOBB.height = trk.tlwh[3];  // height
            trkOBB.angle = trk.tlwh[4];   // angle
            
            for (size_t k = 0; k < obbs.size(); ++k)
            {
                // 使用ProbIoU计算相似度
                float iou = OBBNMSProcessor::calculateProbIOU(trkOBB, obbs[k], false, 1e-7f);
                
                if (iou > bestIou) { 
                    bestIou = iou; 
                    bestIdx = int(k); 
                }
            }
            
            if (bestIdx >= 0 && bestIou > 0.1f) { // 设置一个最小IoU阈值
                tid2obb[trk.track_id] = obbs[bestIdx];
            }
        }

        /* 保存检测图/标签 */
        const cv::Scalar colors[] = {
            {255,0,0},{0,255,0},{0,0,255},{255,255,0},{255,0,255},{0,255,255}
        };
        cv::Mat detImg = img.clone();
        const auto& labels = detector.getLabels();
        for (const auto& o : obbs) {
            std::string label = (o.classIndex < labels.size()) ? 
                               labels[o.classIndex] : "unknown";
            std::string text = label + ":" + std::to_string(o.confidence).substr(0, 4);
            drawOBB(detImg, o, colors[o.classIndex % 6], text);
        }
        cv::imwrite(str_format("%s/%06d.jpg", config.imgOut.c_str(), no), detImg);

        std::ofstream folab(str_format("%s/%06d.txt", config.labelOut.c_str(), no));
        for (const auto& o : obbs)
        {
            auto pts = o.getCornerPoints();
            folab << o.classIndex << ' ' << o.confidence;
            for (const auto& p : pts) folab << ' ' << p.x << ' ' << p.y;
            folab << '\n';
        }

        /* 保存跟踪图（可选） */
        if (!config.trackImgOut.empty())
        {
            cv::Mat trkImg = img.clone();
            for (const auto& trk : stracks)
            {
                cv::Scalar color = tracker.get_color(trk.track_id);
                int cx = int(trk.tlwh[0]);  // 中心点x
                int cy = int(trk.tlwh[1]);  // 中心点y
                cv::putText(trkImg, cv::format("%d", trk.track_id),
                            cv::Point(cx - 10, cy), 0, 0.6, color, 2, cv::LINE_AA);
                            
                auto it = tid2obb.find(trk.track_id);
                if (it != tid2obb.end()) {
                    std::string label = (it->second.classIndex < labels.size()) ? 
                                       labels[it->second.classIndex] : "unknown";
                    std::string text = "ID:" + std::to_string(trk.track_id) + " " + label;
                    drawOBB(trkImg, it->second, color, text);
                } else {
                    // 如果没有关联到检测结果，直接用跟踪结果画框
                    OBBBoundingBox trkOBB;
                    trkOBB.cx = trk.tlwh[0];      // cx
                    trkOBB.cy = trk.tlwh[1];      // cy
                    trkOBB.width = trk.tlwh[2];   // width
                    trkOBB.height = trk.tlwh[3];  // height
                    trkOBB.angle = trk.tlwh[4];   // angle
                    drawOBBFromRotatedRect(trkImg, trkOBB, color, cv::format("ID:%d", trk.track_id));
                }
            }
            cv::imwrite(str_format("%s/%06d.jpg", config.trackImgOut.c_str(), no), trkImg);
        }
    }

    if (frameCnt == 0) { std::cerr << "No frame processed.\n"; return; }

    double fpsDet = frameCnt * 1'000'000.0 / totalDetUs;
    double fpsTrk = frameCnt * 1'000'000.0 / totalTrkUs;
    std::cout << "[Track] 检测模块平均 FPS = " << fpsDet
              << "  |  跟踪模块平均 FPS = " << fpsTrk << '\n';
}

/* ----------------------------------------------------------
 * main
 * ---------------------------------------------------------- */
int main(int argc, char** argv)
{
    if (argc != 2) {
        printUsage(argv[0]);
        return -1;
    }

    Config config;
    if (!config.loadFromFile(argv[1])) {
        std::cerr << "Failed to load config from: " << argv[1] << std::endl;
        return -1;
    }

    std::cout << "Loaded config:\n";
    std::cout << "  Task: " << config.task << "\n";
    std::cout << "  Model: " << config.modelPath << "\n";
    std::cout << "  Model Type: " << config.modelType << "\n";
    std::cout << "  Input: " << config.imageIn << "\n";
    std::cout << "  Class num: " << config.classNum << "\n";
    std::cout << "  Labels: ";
    for (size_t i = 0; i < config.classLabels.size(); ++i) {
        std::cout << config.classLabels[i];
        if (i < config.classLabels.size() - 1) std::cout << ", ";
    }
    std::cout << "\n\n";

    if (config.task == "detect") runDetect(config);
    else if (config.task == "track") runTrack(config);
    else {
        std::cerr << "Invalid task: " << config.task << ". Must be 'detect' or 'track'.\n";
        return -1;
    }
    
    return 0;
}