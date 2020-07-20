#include <torch/torch.h>
#include <iostream>
#include <iomanip>

int main(int argc, char ** argv) {
    auto cuda_available = torch::cuda::is_available();
    torch::Device device(cuda_available ? torch::kCUDA : torch::kCPU);
    std::cout << (cuda_available ? "CUDA OK. Training GPU." : "Training CPU.") << '\n';

    const int64_t input_size = 784;
    const int64_t num_classes = 10;
    const int64_t batch_size = 100;
    const size_t num_epochs = 5;
    const double learning_rate = 0.001;

    const std::string MNIST_data_path = argv[1];
    auto train_dataset = torch::data::datasets::MNIST(MNIST_data_path)
        .map(torch::data::transforms::Normalize<>(0.1307, 0.3081))
        .map(torch::data::transforms::Stack<>());
    auto num_train_samples = train_dataset.size().value();
    auto test_dataset = torch::data::datasets::MNIST(MNIST_data_path, torch::data::datasets::MNIST::Mode::kTest)
        .map(torch::data::transforms::Normalize<>(0.1307, 0.3081))
        .map(torch::data::transforms::Stack<>());
    auto num_test_samples = test_dataset.size().value();

    auto train_loader = torch::data::make_data_loader<torch::data::samplers::RandomSampler>(
        std::move(train_dataset), batch_size);

    auto test_loader = torch::data::make_data_loader<torch::data::samplers::SequentialSampler>(
        std::move(test_dataset), batch_size);

    torch::nn::Linear model(input_size, num_classes);
    model->to(device);
    torch::optim::SGD optimizer(model->parameters(), torch::optim::SGDOptions(learning_rate));
    std::cout << std::fixed << std::setprecision(4);
    std::cout << "training:\n";
    for (size_t epoch = 0; epoch != num_epochs; ++epoch) {
        double running_loss = 0.0;
        size_t num_correct = 0;

        for (auto& batch : *train_loader) {
            auto data = batch.data.view({batch_size, -1}).to(device);
            auto target = batch.target.to(device);

            //forward
            auto output = model->forward(data);
            auto loss = torch::nn::functional::cross_entropy(output, target);
            running_loss += loss.item<double>() * data.size(0);
            auto prediction = output.argmax(1);
            num_correct += prediction.eq(target).sum().item<int64_t>();

            optimizer.zero_grad();
            loss.backward();
            optimizer.step();
        }

        auto sample_mean_loss = running_loss / num_train_samples;
        auto accuracy = static_cast<double>(num_correct) / num_train_samples;

        std::cout << "Epoch [" << (epoch + 1) << "/" << num_epochs << "], Trainset - Loss: "
            << sample_mean_loss << ", Accuracy: " << accuracy << '\n';
    }

    std::cout << "end!\n\n";
    std::cout << "testing:\n";

    model->eval();
    torch::NoGradGuard no_grad;

    double running_loss = 0.0;
    size_t num_correct = 0;

    for (const auto& batch : *test_loader) {
        auto data = batch.data.view({batch_size, -1}).to(device);
        auto target = batch.target.to(device);

        auto output = model->forward(data);

        auto loss = torch::nn::functional::cross_entropy(output, target);

        running_loss += loss.item<double>() * data.size(0);

        auto prediction = output.argmax(1);

        num_correct += prediction.eq(target).sum().item<int64_t>();
    }

    std::cout << "end!\n";

    auto test_accuracy = static_cast<double>(num_correct) / num_test_samples;
    auto test_sample_mean_loss = running_loss / num_test_samples;

    std::cout << "Testset - Loss: " << test_sample_mean_loss << ", Accuracy: " << test_accuracy << '\n';
}
