#include <torch/torch.h>
#include <iostream>

int main(int argc, char ** argv) {
    const int64_t input_size = 1;
    const int64_t output_size = 1;
    const size_t num_epochs = 60;
    const double learning_rate = 0.001;

    // dataset
    auto x_train = torch::randint(0, 10, {15, 1});
    auto y_train = torch::randint(0, 10, {15, 1});

    torch::nn::Linear model(input_size, output_size);
    torch::optim::SGD optimizer(model->parameters(), torch::optim::SGDOptions(learning_rate));
    // floating precision out
    std::cout << std::fixed << std::setprecision(4);
    std::cout << "training:\n";

    for (size_t epoch = 0; epoch != num_epochs; ++epoch) {
        auto output = model(x_train);
        auto loss = torch::nn::functional::mse_loss(output, y_train);
        optimizer.zero_grad();
        loss.backward();
        optimizer.step();

        if ((epoch + 1) % 5 == 0) {
            std::cout << "Epoch [" << (epoch + 1) << "/" << num_epochs <<
                "], Loss: " << loss.item<double>() << "\n";
        }
    }
    std::cout << "end\n";
}
