#include "custom_data.h"
#include <opencv2/opencv.hpp>

#include <torch/torch.h>
#include <torch/script.h>

#include <cmath>
#include <cstdio>
#include <iostream>
#include <sys/types.h>
#include <dirent.h>

const int64_t noise_size                    = 100;
const int64_t batch_size                    = 64;
const int64_t number_epochs                 = 30;
const int64_t checkpoint_every              = 200;
const int64_t number_samples_x_checkpoint   = 10;
const bool restore_checkpoint               = false;
const int64_t log_interval                  = 10;
std::string dataset_folder                  = "./data/custom_data/"; // "./data/data-arts/";
std::string output_folder                   = "./checkpoint/";

using namespace torch;

struct DCGANGeneratorImpl : nn::Module {
  DCGANGeneratorImpl(int noise_size)
      : conv1(nn::ConvTranspose2dOptions(noise_size, 256, 4)
                  .bias(false)),
        batch_norm1(256),
        conv2(nn::ConvTranspose2dOptions(256, 128, 3)
                  .stride(2)
                  .padding(1)
                  .bias(false)),
        batch_norm2(128),
        conv3(nn::ConvTranspose2dOptions(128, 64, 4)
                  .stride(2)
                  .padding(1)
                  .bias(false)),
        batch_norm3(64),
        conv4(nn::ConvTranspose2dOptions(64, 1, 4)
                  .stride(2)
                  .padding(1)
                  .bias(false))
 {
   register_module("conv1", conv1);
   register_module("conv2", conv2);
   register_module("conv3", conv3);
   register_module("conv4", conv4);
   register_module("batch_norm1", batch_norm1);
   register_module("batch_norm2", batch_norm2);
   register_module("batch_norm3", batch_norm3);
 }

 torch::Tensor forward(torch::Tensor x) {
   x = torch::relu(batch_norm1(conv1(x)));
   x = torch::relu(batch_norm2(conv2(x)));
   x = torch::relu(batch_norm3(conv3(x)));
   x = torch::tanh(conv4(x));
   return x;
 }

 nn::ConvTranspose2d conv1, conv2, conv3, conv4;
 nn::BatchNorm2d batch_norm1, batch_norm2, batch_norm3;
};

TORCH_MODULE(DCGANGenerator);

std::vector<std::pair<std::string,int>> list_files( const char* path ) {
   std::vector<std::pair<std::string,int>> paths;
   DIR* dir = opendir( path );
   if ( dir )  {
      struct dirent* hfile;
      errno = 0;
      while (( hfile = readdir( dir )) != NULL ) {
         if ( !strcmp( hfile->d_name, "."  )) continue;
         if ( !strcmp( hfile->d_name, ".." )) continue;
         if ( hfile->d_name[0] == '.') continue;

         if ( strstr( hfile->d_name, ".jpg" )) {
            //printf( "%s%s\n", path,hfile->d_name );
            paths.push_back(std::pair<std::string,int>(string(path)+string(hfile->d_name),0));
         }
      } 
      closedir( dir );
   }
   return paths;
}

int main(int argc, const char* argv[]) {
  if(argc!=2) {
    std::cerr << "./bin <dataset-folder>" << std::endl;
    return 1;
  }
  dataset_folder = argv[1];
  torch::manual_seed(1);
  torch::Device device(torch::kCPU);
  if (torch::cuda::is_available()) {
    std::cout << "CUDA OK Training GPU" << std::endl;
    device = torch::Device(torch::kCUDA);
  }

  DCGANGenerator generator(noise_size);
  generator->to(device);


  nn::Sequential discriminator(
      // Layer 1
      nn::Conv2d(nn::Conv2dOptions(1, 64, 4).stride(2).padding(1).bias(false)),
      nn::LeakyReLU(nn::LeakyReLUOptions().negative_slope(0.2)),
      // Layer 2
      nn::Conv2d(nn::Conv2dOptions(64, 128, 4).stride(2).padding(1).bias(false)),
      nn::BatchNorm2d(128),
      nn::LeakyReLU(nn::LeakyReLUOptions().negative_slope(0.2)),
      // Layer 3
      nn::Conv2d(nn::Conv2dOptions(128, 256, 4).stride(2).padding(1).bias(false)),
      nn::BatchNorm2d(256),
      nn::LeakyReLU(nn::LeakyReLUOptions().negative_slope(0.2)),
      // Layer 4
      nn::Conv2d(nn::Conv2dOptions(256, 1, 3).stride(1).padding(0).bias(false)), nn::Sigmoid()
  );
  discriminator->to(device);


  std::vector<std::pair<std::string,int>> list_data = list_files( dataset_folder.c_str() );

  std::cerr << "SIZE DATA: " << list_data.size() << std::endl;
  /* MNIST DATASET */
  /*auto dataset = torch::data::datasets::MNIST(dataset_folder)
                     .map(torch::data::transforms::Normalize<>(0.5, 0.5))
                     .map(torch::data::transforms::Stack<>());*/

  /* CUSTOM DATASET view: custom_data.h */
  auto dataset = CustomDataset(list_data, 28).map(torch::data::transforms::Stack<>());
  std::cerr << "OK DATA" << std::endl;

  const int64_t batches_per_epoch = std::ceil(dataset.size().value() / static_cast<double>(batch_size));
  auto data_loader = torch::data::make_data_loader(std::move(dataset),torch::data::DataLoaderOptions().batch_size(batch_size).workers(2));
  torch::optim::Adam generator_optimizer(generator->parameters(), torch::optim::AdamOptions(2e-4).betas(std::make_tuple (0.5, 0.5)));
  torch::optim::Adam discriminator_optimizer(discriminator->parameters(), torch::optim::AdamOptions(2e-4).betas(std::make_tuple (0.5, 0.5)));

  if (restore_checkpoint) {
    torch::load(generator, "generator-checkpoint.pt");
    torch::load(generator_optimizer, "generator-optimizer-checkpoint.pt");
    torch::load(discriminator, "discriminator-checkpoint.pt");
    torch::load(discriminator_optimizer, "discriminator-optimizer-checkpoint.pt");
  }

  auto denorm = [] (torch::Tensor tensor) { return tensor.add(1).div_(2).clamp_(0, 1); };
  std::string ultimo_sample = "";
  int64_t checkpoint_counter = 1;
  for (int64_t epoch = 1; epoch <= number_epochs; ++epoch) {
    int64_t batch_index = 0;
    for (torch::data::Example<>& batch : *data_loader) {
      // Train discriminator with real images.
      discriminator->zero_grad();
      torch::Tensor real_images = batch.data.to(device);
      torch::Tensor real_labels = torch::empty(batch.data.size(0), device).uniform_(0.8, 1.0);
      torch::Tensor real_output = discriminator->forward(real_images);
      torch::Tensor d_loss_real = torch::binary_cross_entropy(real_output, real_labels);
      d_loss_real.backward();

      // Train discriminator with fake images.
      torch::Tensor noise       = torch::randn({batch.data.size(0), noise_size, 1, 1}, device);
      torch::Tensor fake_images = generator->forward(noise);
      torch::Tensor fake_labels = torch::zeros(batch.data.size(0), device);
      torch::Tensor fake_output = discriminator->forward(fake_images.detach());
      torch::Tensor d_loss_fake = torch::binary_cross_entropy(fake_output, fake_labels);
      d_loss_fake.backward();

      //std::cerr << fake_images << std::endl;

      torch::Tensor d_loss = d_loss_real + d_loss_fake;
      discriminator_optimizer.step();

      // Train generator.
      generator->zero_grad();
      fake_labels.fill_(1);
      fake_output = discriminator->forward(fake_images);
      torch::Tensor g_loss = torch::binary_cross_entropy(fake_output, fake_labels);
      g_loss.backward();
      generator_optimizer.step();
      batch_index++;
      if (batch_index % log_interval == 0) {
        std::printf(
            "\r[%2ld/%2ld][%3ld/%3ld] D_loss: %.4f | G_loss: %.4f\n",
            epoch,
            number_epochs,
            batch_index,
            batches_per_epoch,
            d_loss.item<float>(),
            g_loss.item<float>());
      }

      if (batch_index % checkpoint_every == 0) {
        string gen_str      = output_folder+"generator-checkpoint.pt";
        string gen_op_str   = output_folder+"generator-optimizer-checkpoint.pt";
        string discr_str    = output_folder+"discriminator-checkpoint.pt";
        string discr_op_str = output_folder+"discriminator-optimizer-checkpoint.pt";
        string sample_str   = output_folder+"dcgan-sample-"+to_string(checkpoint_counter)+".pt";

        torch::Tensor samples = generator->forward(torch::randn({number_samples_x_checkpoint, noise_size, 1, 1}, device));

        torch::save(generator,               gen_str.c_str());
        torch::save(generator_optimizer,     gen_op_str.c_str());
        torch::save(discriminator,           discr_str.c_str());
        torch::save(discriminator_optimizer, discr_op_str.c_str());
        torch::save((samples + 1.0) / 2.0,   sample_str.c_str());
        std::cout << "\n-> checkpoint " << ++checkpoint_counter << '\n';
	ultimo_sample = sample_str;

        fake_images = denorm(fake_images.reshape({fake_images.size(0), 1, 28, 28}));
        //fake_images = fake_images.squeeze().detach().permute({1, 2, 0});
        fake_images = fake_images.mul(255).clamp(0, 255).to(torch::kU8);
        fake_images = fake_images.to(torch::kCPU);

        cv::Mat frame(cv::Size( fake_images.size(2),  fake_images.size(3) ), CV_8UC1, fake_images.data_ptr() );

        if(!frame.empty()) {
          cv::resize(frame,frame,cv::Size(0,0),8.0,8.0);
	  //std::string str = output_folder+"fake_images_"+to_string(epoch)+".png";
	  //std::cerr << str << std::endl;
	  //imwrite(str, frame);
          cv::imshow("frame", frame);
          cv::waitKey(1);
	}
      }
    }
  }
}
