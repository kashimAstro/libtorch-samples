#include <opencv2/opencv.hpp>

#include <torch/torch.h>
#include <vector>

class CustomDataset : public torch::data::Dataset<CustomDataset> {
private:
  std::vector<torch::Tensor> images; 
  std::vector<torch::Tensor> labels;
public:
  // Constructor
  CustomDataset(std::vector<std::pair<std::string,int>> list_data, int image_size) {
    images.resize(list_data.size());
    labels.resize(list_data.size());
    for(int i = 0; i < list_data.size(); i++) {
       cv::Mat mat = cv::imread(list_data[i].first, cv::IMREAD_GRAYSCALE);
       if(mat.empty())
	  std::cerr << "errore open: " << list_data[i].first << std::endl;
       
       cv::resize(mat, mat, cv::Size(image_size, image_size));
       //std::vector<cv::Mat> channels(3);
       //cv::split(mat, channels);

       //auto r = torch::from_blob(channels[2].ptr(), {image_size, image_size}, torch::kUInt8);
       //auto g = torch::from_blob(channels[1].ptr(), {image_size, image_size}, torch::kUInt8);
       //auto b = torch::from_blob(channels[0].ptr(), {image_size, image_size}, torch::kUInt8);

       //auto tensor_image = torch::cat({r, g, b}).view({3, image_size, image_size}).to(torch::kFloat);
       cv::Mat img_float;
       mat.convertTo(img_float, CV_32F);
       auto tensor_image = torch::zeros({ mat.channels(), image_size, image_size });
       memcpy(tensor_image.data_ptr(), img_float.data, tensor_image.numel() * sizeof(float));
       auto tensor_label = torch::from_blob(&list_data[i].second, {1}, torch::kInt);

       images[i] = tensor_image;
       labels[i] = tensor_label; 
    }
  }

  // Override get() function to return tensor at location index
  torch::data::Example<> get(size_t index) override {
    torch::Tensor sample_img = images.at(index);
    torch::Tensor sample_label = labels.at(index);
    return {sample_img.clone(), sample_label.clone()};
  }

  // Return the length of data
  torch::optional<size_t> size() const override {
    return labels.size();
  }
};
