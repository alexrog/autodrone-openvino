#include "nanodet_openvino.h"
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <librealsense2/rs.hpp>
#include "cv-helpers.hpp"
#include <iostream>
#include <sstream>

struct object_rect {
    int x;
    int y;
    int width;
    int height;
};

int resize_uniform(cv::Mat& src, cv::Mat& dst, cv::Size dst_size, object_rect& effect_area)
{
    int w = src.cols;
    int h = src.rows;
    int dst_w = dst_size.width;
    int dst_h = dst_size.height;
    //std::cout << "src: (" << h << ", " << w << ")" << std::endl;
    dst = cv::Mat(cv::Size(dst_w, dst_h), CV_8UC3, cv::Scalar(0));

    float ratio_src = w * 1.0 / h;
    float ratio_dst = dst_w * 1.0 / dst_h;

    int tmp_w = 0;
    int tmp_h = 0;
    if (ratio_src > ratio_dst) {
        tmp_w = dst_w;
        tmp_h = floor((dst_w * 1.0 / w) * h);
    }
    else if (ratio_src < ratio_dst) {
        tmp_h = dst_h;
        tmp_w = floor((dst_h * 1.0 / h) * w);
    }
    else {
        cv::resize(src, dst, dst_size);
        effect_area.x = 0;
        effect_area.y = 0;
        effect_area.width = dst_w;
        effect_area.height = dst_h;
        return 0;
    }

    //std::cout << "tmp: (" << tmp_h << ", " << tmp_w << ")" << std::endl;
    cv::Mat tmp;
    cv::resize(src, tmp, cv::Size(tmp_w, tmp_h));

    if (tmp_w != dst_w) {
        int index_w = floor((dst_w - tmp_w) / 2.0);
        //std::cout << "index_w: " << index_w << std::endl;
        for (int i = 0; i < dst_h; i++) {
            memcpy(dst.data + i * dst_w * 3 + index_w * 3, tmp.data + i * tmp_w * 3, tmp_w * 3);
        }
        effect_area.x = index_w;
        effect_area.y = 0;
        effect_area.width = tmp_w;
        effect_area.height = tmp_h;
    }
    else if (tmp_h != dst_h) {
        int index_h = floor((dst_h - tmp_h) / 2.0);
        //std::cout << "index_h: " << index_h << std::endl;
        memcpy(dst.data + index_h * dst_w * 3, tmp.data, tmp_w * tmp_h * 3);
        effect_area.x = 0;
        effect_area.y = index_h;
        effect_area.width = tmp_w;
        effect_area.height = tmp_h;
    }
    else {
        printf("error\n");
    }
    //cv::imshow("dst", dst);
    //cv::waitKey(0);
    return 0;
}

const int color_list[80][3] =
{
    //{255 ,255 ,255}, //bg
    {216 , 82 , 24},
    {236 ,176 , 31},
    {125 , 46 ,141},
    {118 ,171 , 47},
    { 76 ,189 ,237},
    {238 , 19 , 46},
    { 76 , 76 , 76},
    {153 ,153 ,153},
    {255 ,  0 ,  0},
    {255 ,127 ,  0},
    {190 ,190 ,  0},
    {  0 ,255 ,  0},
    {  0 ,  0 ,255},
    {170 ,  0 ,255},
    { 84 , 84 ,  0},
    { 84 ,170 ,  0},
    { 84 ,255 ,  0},
    {170 , 84 ,  0},
    {170 ,170 ,  0},
    {170 ,255 ,  0},
    {255 , 84 ,  0},
    {255 ,170 ,  0},
    {255 ,255 ,  0},
    {  0 , 84 ,127},
    {  0 ,170 ,127},
    {  0 ,255 ,127},
    { 84 ,  0 ,127},
    { 84 , 84 ,127},
    { 84 ,170 ,127},
    { 84 ,255 ,127},
    {170 ,  0 ,127},
    {170 , 84 ,127},
    {170 ,170 ,127},
    {170 ,255 ,127},
    {255 ,  0 ,127},
    {255 , 84 ,127},
    {255 ,170 ,127},
    {255 ,255 ,127},
    {  0 , 84 ,255},
    {  0 ,170 ,255},
    {  0 ,255 ,255},
    { 84 ,  0 ,255},
    { 84 , 84 ,255},
    { 84 ,170 ,255},
    { 84 ,255 ,255},
    {170 ,  0 ,255},
    {170 , 84 ,255},
    {170 ,170 ,255},
    {170 ,255 ,255},
    {255 ,  0 ,255},
    {255 , 84 ,255},
    {255 ,170 ,255},
    { 42 ,  0 ,  0},
    { 84 ,  0 ,  0},
    {127 ,  0 ,  0},
    {170 ,  0 ,  0},
    {212 ,  0 ,  0},
    {255 ,  0 ,  0},
    {  0 , 42 ,  0},
    {  0 , 84 ,  0},
    {  0 ,127 ,  0},
    {  0 ,170 ,  0},
    {  0 ,212 ,  0},
    {  0 ,255 ,  0},
    {  0 ,  0 , 42},
    {  0 ,  0 , 84},
    {  0 ,  0 ,127},
    {  0 ,  0 ,170},
    {  0 ,  0 ,212},
    {  0 ,  0 ,255},
    {  0 ,  0 ,  0},
    { 36 , 36 , 36},
    { 72 , 72 , 72},
    {109 ,109 ,109},
    {145 ,145 ,145},
    {182 ,182 ,182},
    {218 ,218 ,218},
    {  0 ,113 ,188},
    { 80 ,182 ,188},
    {127 ,127 ,  0},
};

cv::Mat draw_bboxes(const cv::Mat& bgr, const std::vector<BoxInfo>& bboxes, object_rect effect_roi)
{
    static const char* class_names[] = { "rc_car", "person", "car" };

    cv::Mat image = bgr.clone();
    int src_w = image.cols;
    int src_h = image.rows;
    int dst_w = effect_roi.width;
    int dst_h = effect_roi.height;
    float width_ratio = (float)src_w / (float)dst_w;
    float height_ratio = (float)src_h / (float)dst_h;

    for (size_t i = 0; i < bboxes.size(); i++)
    {
        const BoxInfo& bbox = bboxes[i];
	if(bbox.label != 0)
	    continue;
        cv::Scalar color = cv::Scalar(color_list[bbox.label][0], color_list[bbox.label][1], color_list[bbox.label][2]);
        //fprintf(stderr, "%d = %.5f at %.2f %.2f %.2f %.2f\n", bbox.label, bbox.score,
        //    bbox.x1, bbox.y1, bbox.x2, bbox.y2);

        cv::rectangle(image, cv::Rect(cv::Point((bbox.x1 - effect_roi.x) * width_ratio, (bbox.y1 - effect_roi.y) * height_ratio),
                                      cv::Point((bbox.x2 - effect_roi.x) * width_ratio, (bbox.y2 - effect_roi.y) * height_ratio)), color);

        char text[256];
        sprintf(text, "%s %.1f%%", class_names[bbox.label], bbox.score * 100);


        int baseLine = 0;
        cv::Size label_size = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.4, 1, &baseLine);

        int x = (bbox.x1 - effect_roi.x) * width_ratio;
        int y = (bbox.y1 - effect_roi.y) * height_ratio - label_size.height - baseLine;
        if (y < 0)
            y = 0;
        if (x + label_size.width > image.cols)
            x = image.cols - label_size.width;

        cv::rectangle(image, cv::Rect(cv::Point(x, y), cv::Size(label_size.width, label_size.height + baseLine)),
            color, -1);

	cv::Rect boundingbox = cv::Rect(cv::Point(x, y), cv::Size(label_size.width, label_size.height + baseLine));
 
	if(bbox.label == 0)
	    std::cout << "[(" << boundingbox.x << "," << boundingbox.y << "),("<< boundingbox.x + boundingbox.width << "," << boundingbox.y << "),(" << boundingbox.x << "," << boundingbox.y + boundingbox.height << "),(" << boundingbox.x + boundingbox.width << "," << boundingbox.y + boundingbox.height << ")]" << std::endl;

        cv::putText(image, text, cv::Point(x, y + label_size.height),
            cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(255, 255, 255));
    }

    // cv::imshow("image", image);
    // cv::imwrite("image.jpg", image);
    // std::cout << "saved image" << std::endl;

    return image;
}


int image_inference(NanoDet& detector, const char* imagepath)
{
    // const char* imagepath = "D:/Dataset/coco/val2017/*.jpg";
    
    std::vector<std::string> filenames;
    cv::glob(imagepath, filenames, false);
    int height = detector.input_size[0];
    int width = detector.input_size[1];

    for (auto img_name : filenames)
    {
        cv::Mat image = cv::imread(img_name);
        if (image.empty())
        {
            fprintf(stderr, "cv::imread failed\n");
            return -1;
        }
        object_rect effect_roi;
        cv::Mat resized_img;
        resize_uniform(image, resized_img, cv::Size(width, height), effect_roi);
        auto results = detector.detect(resized_img, 0.4, 0.5);
        
        image = draw_bboxes(image, results, effect_roi);
        //cv::waitKey(0);

    }
    return 0;
}

int intelrealsense_inference(NanoDet& detector)
{
    using namespace cv;
    using namespace rs2;

    const size_t inWidth      = 512;
    const size_t inHeight     = 288;
    const float WHRatio       = inWidth / (float)inHeight;
    const float inScaleFactor = 0.007843f;
    const float meanVal       = 127.5;
    int height = detector.input_size[0];
    int width = detector.input_size[1];

    // Start streaming from Intel RealSense Camera
    pipeline pipe;
    auto config = pipe.start();
    auto profile = config.get_stream(RS2_STREAM_COLOR)
                         .as<video_stream_profile>();
    rs2::align align_to(RS2_STREAM_COLOR);

    Size cropSize;
    if (profile.width() / (float)profile.height() > WHRatio)
    {
        cropSize = Size(static_cast<int>(profile.height() * WHRatio),
                        profile.height());
    }
    else
    {
        cropSize = Size(profile.width(),
                        static_cast<int>(profile.width() / WHRatio));
    }

    Rect crop(Point((profile.width() - cropSize.width) / 2,
                    (profile.height() - cropSize.height) / 2),
              cropSize);

    const auto window_name = "Display Image";
    namedWindow(window_name, WINDOW_AUTOSIZE);

    while (getWindowProperty(window_name, WND_PROP_AUTOSIZE) >= 0)
    {
	auto start = std::chrono::steady_clock::now();
        // Wait for the next set of frames
        auto data = pipe.wait_for_frames();

        auto color_frame = data.get_color_frame();

        // Convert RealSense frame to OpenCV matrix:
        auto color_mat = frame_to_mat(color_frame);

        cv::Mat resized_img;
        object_rect effect_roi;
        resize_uniform(color_mat, resized_img, cv::Size(width, height), effect_roi);
        auto results = detector.detect(resized_img, 0.4, 0.5);
        cv::Mat image = draw_bboxes(color_mat, results, effect_roi);

        auto end = std::chrono::steady_clock::now();
        double time = std::chrono::duration<double, std::milli>(end - start).count();
        double fps = 1/ (time/1000);
        std::stringstream stream;
        stream << std::fixed << std::setprecision(2) << fps;
        std::string s = stream.str();
        fps = std::ceil(fps * 100.0) / 100.0;
        cv::putText(image, s, cv::Point(30,100), cv::FONT_HERSHEY_SIMPLEX,2.1,cv::Scalar(0,0,255), 2, cv::LINE_AA);
            imshow(window_name, image);
            if (waitKey(1) >= 0) break;
        }
        return 0;
}

int video_inference(NanoDet& detector, const char* path) {
    cv::Mat image;
    cv::VideoCapture* cap = new cv::VideoCapture(path);

    int height = detector.input_size[0];
    int width = detector.input_size[1];
    // get number of frames in video
    int nFrames = cap->get(cv::CAP_PROP_FRAME_COUNT);
    int orig_fps = cap->get(cv::CAP_PROP_FPS);
    fprintf(stderr, "Num frames: %d FPS: %d\n", nFrames, orig_fps);
    
    // create video writer
    int frame_width = cap->get(cv::CAP_PROP_FRAME_WIDTH);
    int frame_height = cap->get(cv::CAP_PROP_FRAME_HEIGHT);
    cv::VideoWriter video("../outcpp.avi", cv::VideoWriter::fourcc('M','J','P','G'), orig_fps, cv::Size(frame_width,frame_height));

    auto start_full_time = std::chrono::steady_clock::now();
    double decoding_time = 0;
    double inferencing_time = 0;
    int i = 0;
    while (true)
    {
        auto start_decoding = std::chrono::steady_clock::now();

        *cap >> image;
        if(image.empty()) {
            break;
        }
        object_rect effect_roi;
        cv::Mat resized_img;
        resize_uniform(image, resized_img, cv::Size(width, height), effect_roi);

        auto end_decoding = std::chrono::steady_clock::now();
        decoding_time += std::chrono::duration<double, std::milli>(end_decoding - start_decoding).count();

        auto start_inferencing = std::chrono::steady_clock::now();

        auto results = detector.detect(resized_img, 0.4, 0.5);
        cv::Mat image_new = draw_bboxes(image, results, effect_roi);

        auto end_inferencing = std::chrono::steady_clock::now();
        inferencing_time += std::chrono::duration<double, std::milli>(end_inferencing - start_inferencing).count();
        video.write(image_new);
        //cv::waitKey(1);
    }

    auto end = std::chrono::steady_clock::now();
    double time = std::chrono::duration<double, std::milli>(end - start_full_time).count();

    std::cout << "total time: " << time/1000 << "s" << std::endl;
    std::cout << "decoding time : " << decoding_time/1000 << "s" <<std::endl;
    std::cout << "inference time: " << inferencing_time/1000 << "s" << std::endl;
    std::cout << "total fps: " << nFrames / (time/1000) << std::endl;

    cap->release();
    video.release();

    return 0; 
}

int main(int argc, char** argv)
{
    std::string mode = "live";
    std::string device = "CPU";
    std::string model_path = "nanodet.xml";
    std::string img_vid_path = "";
    int precision = 32;
    for(int i = 1; i < argc - 1; i++) {
        std::string curr(argv[i]);
        std::string next(argv[i+1]);
        if(curr == "--mode" || curr == "-m") {
            mode = next;
            i++;
        }
        else if(curr == "--device" || curr == "-d") {
            device = next;
            i++;
        }
        else if(curr == "--model_path" || curr == "-mp") {
            model_path = next;
            i++;
        }
        else if(curr == "--image_path" || curr == "-i" || curr == "--video_path" || curr == "-v") {
            img_vid_path = next;
            i++;
        }
        else if(curr == "--precision" || curr == "-p") {
            if(next == "FP16") {
                precision = 16;
            }
            else if(next == "FP32") {
                precision = 32;
            }
            else {
                fprintf(stderr, "invalid precision: must be of type 'FP16' or 'FP32'\n");
                return -1;
            }
            i++;
        }
        else {
            fprintf(stderr, "invalid flag: %s\n", argv[i]);
        }
    }

    if(!(mode == "live" || mode == "video" || mode == "img")) {
        fprintf(stderr, "invalid mode: must be of type 'live', 'video', or 'img\n");
        return -1;
    }

    std::cout<<"start init model"<<std::endl;
    auto detector = NanoDet(const_cast<char*>(model_path.c_str()), const_cast<char*>(device.c_str()), precision);
    std::cout<<"Current Params:" << std::endl;
    std::cout << "-model_path: " << model_path << std::endl;
    std::cout << "-mode: " << mode << std::endl;
    std::cout << "-device: " << device << std::endl;
    std::cout << "-precision: FP" << precision << std::endl;
    if(mode == "img" || mode == "video") {
        std::cout << "-" <<  mode << "_path: " << img_vid_path << std::endl;
    }

    if(mode == "live") {
        intelrealsense_inference(detector);
    }
    else if(mode == "img") {
        image_inference(detector, const_cast<char*>(img_vid_path.c_str()));
    }
    else if(mode == "video") {
        video_inference(detector, const_cast<char*>(img_vid_path.c_str()));
    }
}
