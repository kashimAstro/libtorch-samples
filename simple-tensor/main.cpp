#include <torch/torch.h>
#include <iostream>
#include <chrono>
#include <time.h>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/imgproc.hpp>

// torch::<function-name>(<function-specific-options>, <sizes>, <tensor-options>)
// https://pytorch.org/cppdocs/notes/tensor_creation.html

using namespace std;

int main(int argc, char ** argv) {
	auto options = torch::TensorOptions().dtype(torch::kFloat32).layout(torch::kStrided).device(torch::kCUDA, 0).requires_grad(true);

        torch::Tensor tensor0 = torch::randint(/*low=*/1, /*high=*/10, {15, 15}, options);
        cerr << tensor0 << endl;

	torch::Tensor tensor1 = torch::ones(10, options);
	cerr << tensor1 << endl;

	torch::Tensor tensor2 = torch::arange(10, 20, options);
	cerr << tensor2 << endl;
        
        torch::Tensor tensor3 = torch::eye(10, 10, options);
	cerr << tensor3 << endl;

        torch::save({tensor0, tensor1, tensor2, tensor3}, "tensor_save.pt");

	cerr << "###################################" << endl;
	cerr << "load .pt tensor" << endl;

	vector<torch::Tensor> r_tensor;
	torch::load(r_tensor, "tensor_save.pt");
	for(int i = 0; i < r_tensor.size(); i++)
	        cerr << r_tensor[i] << endl;
	
	return 0;
}
