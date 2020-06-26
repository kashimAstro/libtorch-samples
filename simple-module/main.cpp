#include <iostream>
#include <torch/torch.h>

struct Model : torch::nn::Module {
    torch::nn::Linear input{nullptr}, hidden{nullptr}, output{nullptr};

    Model() {
        input  = register_module("input",  torch::nn::Linear(8,64));
        hidden = register_module("hidden", torch::nn::Linear(64,64));
        output = register_module("output", torch::nn::Linear(64,1));
    }

    torch::Tensor forward(torch::Tensor X){
        X = torch::relu(input->forward(X));
        X = torch::relu(hidden->forward(X));
        X = torch::sigmoid(output->forward(X));
        return X;
    }
};


int main(int argc, char ** argv) {
    Model model;
    
    auto input  = torch::rand({8,});
    auto output = model.forward(input);

    std::cout << input  << std::endl;
    std::cout << output << std::endl;
    return 0;
}
