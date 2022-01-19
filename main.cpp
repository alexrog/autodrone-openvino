#include "nanodet_openvino.h"
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <librealsense2/rs.hpp>
#include "cv-helpers.hpp"
#include <iostream>
#include <thread>
#include <vector>
#include <chrono>
#include <thread>
#include <pthread.h>
#include <queue>

struct object_rect {
    int x;
    int y;
    int width;
    int height;
};

std::queue<cv::Mat> image_queue;
std::queue<object_rect> effect_roi_queue;
cv::VideoCapture* cap;
bool done_decoding = false;
bool processing_image = false;
auto global_detector = NanoDet("nanodet.xml");

pthread_mutex_t mutex = PTHREAD_MUTEX_INITIALIZER;
 
pthread_cond_t dataNotProduced =
                    PTHREAD_COND_INITIALIZER;
pthread_cond_t dataNotConsumed =
                    PTHREAD_COND_INITIALIZER;



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
    static const char* class_names[] = { "rc_car" };/*"person", "bicycle", "car", "motorcycle", "airplane", "bus",
                                        "train", "truck", "boat", "traffic light", "fire hydrant",
                                        "stop sign", "parking meter", "bench", "bird", "cat", "dog",
                                        "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe",
                                        "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
                                        "skis", "snowboard", "sports ball", "kite", "baseball bat",
                                        "baseball glove", "skateboard", "surfboard", "tennis racket",
                                        "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl",
                                        "banana", "apple", "sandwich", "orange", "broccoli", "carrot",
                                        "hot dog", "pizza", "donut", "cake", "chair", "couch",
                                        "potted plant", "bed", "dining table", "toilet", "tv", "laptop",
                                        "mouse", "remote", "keyboard", "cell phone", "microwave", "oven",
                                        "toaster", "sink", "refrigerator", "book", "clock", "vase",
                                        "scissors", "teddy bear", "hair drier", "toothbrush"
    };*/

    cv::Mat image = bgr.clone();
    int src_w = image.cols;
    int src_h = image.rows;
    int dst_w = effect_roi.width;
    int dst_h = effect_roi.height;
    float width_ratio = (float)src_w / (float)dst_w;
    float height_ratio = (float)src_h / (float)dst_h;

    //fprintf(stderr, "output size: %d\n", bboxes.size());
    for (size_t i = 0; i < bboxes.size(); i++)
    {
        const BoxInfo& bbox = bboxes[i];
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

        cv::putText(image, text, cv::Point(x, y + label_size.height),
            cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(255, 255, 255));
    }

    // cv::imshow("image", image);
    // cv::imwrite("image.jpg", image);
    // std::cout << "saved image" << std::endl;

    return image;
}


int image_demo(NanoDet& detector, const char* imagepath)
{
    // const char* imagepath = "D:/Dataset/coco/val2017/*.jpg";
    
    std::vector<std::string> filenames;
    cv::glob(imagepath, filenames, false);

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
        resize_uniform(image, resized_img, cv::Size(320, 320), effect_roi);
        auto results = detector.detect(resized_img, 0.4, 0.5);
        
        image = draw_bboxes(image, results, effect_roi);
        //cv::waitKey(0);

    }
    return 0;
}

int webcam_demo(NanoDet& detector, int cam_id)
{
    using namespace cv;
    using namespace rs2;

    const size_t inWidth      = 300;
    const size_t inHeight     = 300;
    const float WHRatio       = inWidth / (float)inHeight;
    const float inScaleFactor = 0.007843f;
    const float meanVal       = 127.5;
    //const char* classNames[]  = {"rc_car"};

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
        // Wait for the next set of frames
        auto data = pipe.wait_for_frames();
        // Make sure the frames are spatially aligned
        data = align_to.process(data);

        auto color_frame = data.get_color_frame();
        auto depth_frame = data.get_depth_frame();

        // If we only received new depth frame, 
        // but the color did not update, continue
        static int last_frame_number = 0;
        if (color_frame.get_frame_number() == last_frame_number) continue;
        last_frame_number = static_cast<int>(color_frame.get_frame_number());

        // Convert RealSense frame to OpenCV matrix:
        auto color_mat = frame_to_mat(color_frame);
        auto depth_mat = depth_frame_to_meters(depth_frame);

	cv::Mat resized_img;
	object_rect effect_roi;
        resize_uniform(color_mat, resized_img, cv::Size(320, 320), effect_roi);
        auto results = detector.detect(resized_img, 0.4, 0.5);
        cv::Mat image = draw_bboxes(color_mat, results, effect_roi);

        imshow(window_name, color_mat);
        if (waitKey(1) >= 0) break;
    }
    return 0;
}

void* decode_video(void* args) {
    cv::Mat image;
    done_decoding = false;
    while(true) {    
        if(!done_decoding) {
            pthread_mutex_lock(&mutex);
            *cap >> image;
            if(image.empty()) {
                done_decoding = true;
                fprintf(stderr, "Done decoding: %lu\n", image_queue.size());
                pthread_cond_signal(&dataNotProduced);
                pthread_mutex_unlock(&mutex);
                break;
            }
            
            object_rect effect_roi;
            cv::Mat resized_img;
            resize_uniform(image, resized_img, cv::Size(320, 320), effect_roi); 

            effect_roi_queue.push(effect_roi);
            image_queue.push(resized_img);
            pthread_cond_signal(&dataNotProduced);

            //if(!processing_image)
                pthread_cond_wait(&dataNotConsumed, &mutex);
        }
        else {
            std::cout << ">> Producer is in wait.." << std::endl;
            pthread_cond_wait(&dataNotConsumed, &mutex);
        }

        pthread_mutex_unlock(&mutex);
    }
}

void* perform_inferences(void*) {
    int i = 0;
    while (true) {
        pthread_mutex_lock(&mutex);
 
        // Pop only when queue has at least 1 element
        if (image_queue.size() > 0 && effect_roi_queue.size() > 0) {
            //fprintf(stderr, "%lu\n", image_queue.size());
            // Get the data from the front of queue
            cv::Mat image = image_queue.front();
            object_rect effect_roi = effect_roi_queue.front();

            image_queue.pop();
            effect_roi_queue.pop();
 
            pthread_cond_signal(&dataNotConsumed);
            pthread_mutex_unlock(&mutex);
            processing_image = true;
 
            // cout << "B thread consumed: " << data << endl;
 
            // perform detection
            auto results = global_detector.detect(image, 0.4, 0.5);
            cv::Mat image_new = draw_bboxes(image, results, effect_roi); 
            processing_image = false;
            // std::this_thread::sleep_for(std::chrono::milliseconds(1000));

            // video.write(image_new); 
 
            // Pop the consumed data from queue
        }
 
        // Check if consumed numbers from both threads
        // has reached to MAX value
        else if (done_decoding) {
            pthread_mutex_unlock(&mutex);
            return NULL;
        }
 
        // If some other thread is executing, wait
        else {
            //std::cout << "B is in wait.." << std::endl;
            pthread_cond_wait(&dataNotProduced, &mutex);
            pthread_mutex_unlock(&mutex);
        }
 
        // Get the mutex unlocked
    }
}

int video_demo_multi(NanoDet& detector, const char* path)
{
    cv::Mat image;
    cap = new cv::VideoCapture(path);

    // get number of frames in video
    int nFrames = cap->get(cv::CAP_PROP_FRAME_COUNT);
    int orig_fps = cap->get(cv::CAP_PROP_FPS);
    fprintf(stderr, "Num frames: %d FPS: %d\n", nFrames, orig_fps);
    
    // create video writer
    int frame_width = cap->get(cv::CAP_PROP_FRAME_WIDTH);
    int frame_height = cap->get(cv::CAP_PROP_FRAME_HEIGHT);
    cv::VideoWriter video("../Output/outcpp.avi", cv::VideoWriter::fourcc('M','J','P','G'), orig_fps, cv::Size(frame_width,frame_height));
    
    pthread_t decode_thread, inference_thread;
    int decode_producer = pthread_create(&decode_thread, NULL, decode_video, NULL);

    int inference_producer = pthread_create(&inference_thread, NULL, perform_inferences, NULL);

    auto start_full_time = std::chrono::steady_clock::now();
    if (!decode_producer)
        pthread_join(decode_thread, NULL);

    if (!inference_producer)
        pthread_join(inference_thread, NULL);

    auto end = std::chrono::steady_clock::now();
    double time = std::chrono::duration<double, std::milli>(end - start_full_time).count();

    std::cout << "total time: " << time/1000 << "s" << std::endl;
    std::cout << "total fps: " << nFrames / (time/1000) << std::endl;

    cap->release();
    video.release();

    return 0;
}

int video_demo(NanoDet& detector, const char* path) {
    cv::Mat image;
    cap = new cv::VideoCapture(path);

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
        resize_uniform(image, resized_img, cv::Size(320, 320), effect_roi);

        auto end_decoding = std::chrono::steady_clock::now();
        decoding_time += std::chrono::duration<double, std::milli>(end_decoding - start_decoding).count();

        auto start_inferencing = std::chrono::steady_clock::now();

        auto results = detector.detect(resized_img, 0.4, 0.5);
        cv::Mat image_new = draw_bboxes(image, results, effect_roi);

        auto end_inferencing = std::chrono::steady_clock::now();
        inferencing_time += std::chrono::duration<double, std::milli>(end_inferencing - start_inferencing).count();

	    std::cout << "i: " << i++ << std::endl;
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

int benchmark(NanoDet& detector)
{
    int loop_num = 100;
    int warm_up = 8;

    double time_min = DBL_MAX;
    double time_max = -DBL_MAX;
    double time_avg = 0;
    cv::Mat image(320, 320, CV_8UC3, cv::Scalar(1, 1, 1));

    for (int i = 0; i < warm_up + loop_num; i++)
    {
        auto start = std::chrono::steady_clock::now();
        std::vector<BoxInfo> results;
        results = detector.detect(image, 0.4, 0.5);
        auto end = std::chrono::steady_clock::now();
        double time = std::chrono::duration<double, std::milli>(end - start).count();
        if (i >= warm_up)
        {
            time_min = (std::min)(time_min, time);
            time_max = (std::max)(time_max, time);
            time_avg += time;
        }
    }
    time_avg /= loop_num;
    fprintf(stderr, "%20s  min = %7.2f  max = %7.2f  avg = %7.2f\n", "nanodet", time_min, time_max, time_avg);
    return 0;
}


int main(int argc, char** argv)
{
    if (argc != 3)
    {
        fprintf(stderr, "usage: %s [mode] [path]. \n For webcam mode=0, path is cam id; \n For image demo, mode=1, path=xxx/xxx/*.jpg; \n For video, mode=2; \n For benchmark, mode=3 path=0.\n", argv[0]);
        return -1;
    }
    std::cout<<"start init model"<<std::endl;
    auto detector = NanoDet("nanodet.xml");
    std::cout<<"success"<<std::endl;
    int mode = atoi(argv[1]);
    
    switch (mode)
    {
    case 0:{
        int cam_id = atoi(argv[2]);
        webcam_demo(detector, cam_id);
        break;
        }
    case 1:{
        const char* images = argv[2];
        image_demo(detector, images);
        break;
        }
    case 2:{
        const char* path = argv[2];
        video_demo(detector, path);
        break;
        }
    case 3:{
        benchmark(detector);
        break;
        }
    case 4:{
        const char* path = argv[2];
        video_demo_multi(detector, path);
        break;
    }
    default:{
        fprintf(stderr, "usage: %s [mode] [path]. \n For webcam mode=0, path is cam id; \n For image demo, mode=1, path=xxx/xxx/*.jpg; \n For video, mode=2; \n For benchmark, mode=3 path=0.\n", argv[0]);
        break;
        }
    }
}
