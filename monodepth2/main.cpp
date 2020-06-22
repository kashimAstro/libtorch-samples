#include <iostream>
#include <torch/script.h>
#include <torch/torch.h>
#include <opencv2/opencv.hpp>

using namespace std;

int main(int argc,char**argv) {
    if(argc!=4) return 1;
    string enc = argv[1];
    string dec = argv[2];
    string vid = argv[3];

    torch::jit::script::Module encoder = torch::jit::load(enc);
    encoder.to(at::kCUDA);
    torch::jit::script::Module decoder = torch::jit::load(dec);
    decoder.to(at::kCUDA);
    cv::Mat src;
    cv::Mat input_mat;
    cv::VideoCapture cap(vid);
    int w = 320;
    int h = 240;
    while (1) {
        if(!cap.read(src))
            break;
        cv::resize(src,input_mat,cv::Size(w,h));
        input_mat.convertTo(input_mat,CV_32FC3,1./255.);
        torch::Tensor tensor_image = torch::from_blob(input_mat.data, {1,input_mat.rows, input_mat.cols,3}, torch::kF32);
        tensor_image = tensor_image.permute({0,3,1,2});
        tensor_image = tensor_image.to(at::kCUDA);

        std::vector<torch::IValue> batch;
        batch.push_back(tensor_image);
        auto result_encoder = encoder.forward(batch);
        cout<<*result_encoder.type()<<endl;

        batch.clear();
        batch.push_back(result_encoder);
        auto result_decoder = decoder.forward(batch);
        auto tensor_result = result_decoder.toTensor().to(at::kCPU);
        tensor_result = tensor_result.permute({0,3,2,1});
        cout<<tensor_result.sizes()<<endl;

        cv::Mat disp=cv::Mat(h,w,CV_32FC1,tensor_result.data_ptr());
        cv::resize(disp,disp,cv::Size(src.cols,src.rows));
        disp*=512;

        disp.convertTo(disp,CV_8UC1);
        cv::cvtColor(disp,disp,cv::COLOR_GRAY2BGR);
        src.push_back(disp);
        vector<cv::Mat> channels={disp,disp,disp};
        cv::merge(channels,disp);
        cv::resize(src,src,cv::Size(),0.5,0.5);

        cv::imshow("result",disp);
        cv::imshow("src",src);
        cv::imshow("res",input_mat);

        if(cv::waitKey(1)==27)
            break;
    }

    cout<<"hello"<<endl;
    return 0;
}
