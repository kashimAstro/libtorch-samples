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
   if (argc != 5) {
      std::cerr << "<image> <yolo.cfg> <yolo.weights> <labels.txt>" << std::endl;
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

    vector<string> labels = net.load_labels(argv[4]);
    net.load_weights(argv[3]);
    std::cout << "weight and labels loaded." << endl;
    net.to(device);
    torch::NoGradGuard no_grad;
    net.eval();

    cv::Mat origin_image, resized_image;
    origin_image = cv::imread(argv[1]);
    
    cv::cvtColor(origin_image, resized_image, cv::COLOR_BGR2RGB);
    cv::resize(resized_image, resized_image, cv::Size(input_image_size, input_image_size));
    
    cv::Mat img_float;
    resized_image.convertTo(img_float, CV_32F, 1.0/255);

    std::cout << "start to inference ..." << endl;
    
    auto img_tensor = torch::from_blob(img_float.data, {1, input_image_size, input_image_size, 3}).to(device);
    img_tensor = img_tensor.permute({0,3,1,2});
    
    auto start = std::chrono::high_resolution_clock::now();
    auto output = net.forward(img_tensor);
    
    // class_num               = labels.size
    // confidence              = 0.6
    // non-maximum suppression = 0.4 
    auto result = net.write_results(output, labels, 0.6, 0.4);
    cerr << result << endl;
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = duration_cast<milliseconds>(end - start); 
    std::cout << "inference : " << duration.count() << " ms" << endl; 
    if (result.dim() > 1) {
       int obj_num = result.size(0);
       std::cout << obj_num << " objects found" << endl;
       float w_scale = float(origin_image.cols) / input_image_size;
       float h_scale = float(origin_image.rows) / input_image_size;
       
       result.select(1,1).mul_(w_scale);
       result.select(1,2).mul_(h_scale);
       result.select(1,3).mul_(w_scale);
       result.select(1,4).mul_(h_scale);
       
       auto result_data = result.accessor<float, 2>();
       
       for (int i = 0; i < result.size(0) ; i++) {
          float prob = result_data[i][6];
          int indx = result_data[i][7];
	  string str = "detect: " + labels[indx] + " prob: " + to_string(prob);
          cv::rectangle(origin_image, cv::Point(result_data[i][1], result_data[i][2]), cv::Point(result_data[i][3], result_data[i][4]), cv::Scalar(255, 255, 0), 4, 1, 0);
	  int baseline=0;
	  cv::Size ts = getTextSize(str, cv::FONT_HERSHEY_PLAIN, 1, 2, &baseline);
          cv::Point tp(1,origin_image.rows-ts.height);
          cv::Rect tr(result_data[i][1], result_data[i][2]-ts.height-baseline,ts.width,ts.height*2);
	  cv::rectangle(origin_image, tr, cv::Scalar(255, 255, 255), -1, 1, 0);
	  cv::putText(origin_image, str, cv::Point(result_data[i][1], result_data[i][2]), cv::FONT_HERSHEY_PLAIN, 1, cv::Scalar(0,0,255,255));

          }
       
       cv::imshow("frame", origin_image);
       cv::waitKey(0);
       }
    
    std::cout << "Done" << endl;
    
    return 0;
   }
