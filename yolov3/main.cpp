#include <torch/torch.h>
#include <iostream>
#include <chrono>
#include <time.h>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "darknet.h"

using namespace std; 
using namespace std::chrono; 

int main(int argc, const char* argv[]) {
   if (argc != 4) {
      std::cerr << "<image> <yolo.cfg> <yolo.weights>" << std::endl;
      return -1;
      }

    torch::DeviceType device_type;
    if (torch::cuda::is_available() ) {        
       device_type = torch::kCUDA;
       } else {
       device_type = torch::kCPU;
       }
    torch::Device device(device_type);
    
    int input_image_size = 416;
    darknet_yolo net(argv[2], &device);
    map<string, string> *info = net.get_net_info();
    info->operator[]("height") = std::to_string(input_image_size);
    std::cout << "loading weight ..." << endl;
    net.load_weights(argv[3]);
    std::cout << "weight loaded ..." << endl;
    net.to(device);
    torch::NoGradGuard no_grad;
    net.eval();
    std::cout << "start to inference ..." << endl;
    
    cv::Mat origin_image, resized_image;
    origin_image = cv::imread(argv[1]);
    
    cv::cvtColor(origin_image, resized_image,  cv::COLOR_BGR2RGB);
    cv::resize(resized_image, resized_image, cv::Size(input_image_size, input_image_size));
    
    cv::Mat img_float;
    resized_image.convertTo(img_float, CV_32F, 1.0/255);

    auto img_tensor = torch::from_blob(img_float.data, {1, input_image_size, input_image_size, 3}).to(device);
    img_tensor = img_tensor.permute({0,3,1,2});
    
    auto start = std::chrono::high_resolution_clock::now();
    auto output = net.forward(img_tensor);
    
    // filter result by NMS 
    // class_num = 80
    // confidence = 0.6
    auto result = net.write_results(output, 80, 0.6, 0.4);

    auto end = std::chrono::high_resolution_clock::now();

    auto duration = duration_cast<milliseconds>(end - start); 
    std::cout << "inference taken : " << duration.count() << " ms" << endl; 
    if (result.dim() == 1) {
       std::cout << "no object found" << endl;
       }
    else {
       int obj_num = result.size(0);
       std::cout << obj_num << " objects found" << endl;
       float w_scale = float(origin_image.cols) / input_image_size;
       float h_scale = float(origin_image.rows) / input_image_size;
       
       result.select(1,1).mul_(w_scale);
       result.select(1,2).mul_(h_scale);
       result.select(1,3).mul_(w_scale);
       result.select(1,4).mul_(h_scale);
       
       auto result_data = result.accessor<float, 2>();
       
       for (int i = 0; i < result.size(0) ; i++){
          cv::rectangle(origin_image, cv::Point(result_data[i][1], result_data[i][2]), cv::Point(result_data[i][3], result_data[i][4]), cv::Scalar(0, 0, 255), 1, 1, 0);
          }
       
       cv::imshow("frame", origin_image);
       cv::waitKey(0);
       }
    
    std::cout << "Done" << endl;
    
    return 0;
   }
