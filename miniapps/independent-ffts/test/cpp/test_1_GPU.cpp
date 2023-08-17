#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
// #include <cudalibxt.h>
#include <cufftXt.h>
// #include <driver_types.h>
#include <algorithm>
#include <complex>
#include <cuda_runtime.h>
#include <cufft.h>
#include <cufftmg.hpp>
#include <doctest.h>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <list>
#include <string>
#include <tuple>
#include <vector>

void print_1D(cufftDoubleComplex *array, int size) {
  std::cout << "[";
  for (int i = 0; i < size; i++) {
    std::cout << std::setw(4) << std::setprecision(2) << std::fixed << "("
              << array[i].x << ", " << array[i].y << ")";
    if (i < size - 1) {
      std::cout << ", ";
    }
  }
  std::cout << "]" << std::endl;
}

void print_2D(cufftDoubleComplex *array, int rows, int cols) {
  std::cout << "[" << std::endl;
  for (int i = 0; i < rows; i++) {
    std::cout << " [";
    for (int j = 0; j < cols; j++) {
      std::cout << std::setw(4) << std::setprecision(2) << std::fixed << "("
                << array[i * cols + j].x << ", " << array[i * cols + j].y
                << ")";
      if (j < cols - 1) {
        std::cout << ", ";
      }
    }
    std::cout << "]";
    if (i < rows - 1) {
      std::cout << ", " << std::endl;
    }
  }
  std::cout << std::endl << "]" << std::endl;
}

void read_binary(std::vector<cufftDoubleComplex> &v, std::string filename) {
  std::ifstream input(filename, std::ios::binary | std::ios::in);

  if (!input.is_open()) {
    std::cout << "Error opening data file: " << filename << std::endl;
    exit(-1);
  }

  input.seekg(0, std::ios::end);
  const size_t num_elements = input.tellg() / (2 * sizeof(double));
  input.seekg(0, std::ios::beg);

  std::cout << "Reading " << num_elements << " elements from " << filename
            << std::endl;

  v.resize(num_elements);
  input.read(reinterpret_cast<char *>(v.data()),
             2 * num_elements * sizeof(double));
  input.close();
}

TEST_CASE("FFTHandler: 1 GPU") {

  std::vector<cufftDoubleComplex> h_truth;
  read_binary(h_truth, "../data/40_fft.bin");

  cufftmg::FFTHandler handler = cufftmg::FFTHandler();

  int device_ids[1] = {0};
  int num_devices = 1;
  int N = 40;

  uintptr_t streams[num_devices];
  cudaStream_t stream;
  cudaStreamCreate(&stream);
  streams[0] = (uintptr_t)stream;

  size_t work_sizes[] = {0};

  handler.configure(device_ids, N, num_devices, streams, work_sizes);

  CHECK(handler.num_devices == num_devices);
  CHECK(handler.size == N);
  CHECK(handler.device_ids[0] == device_ids[0]);
  CHECK(handler.device_ids[1] == device_ids[1]);
  CHECK(handler.streams[0] == streams[0]);
  CHECK(handler.streams[1] == streams[1]);
  CHECK(handler.work_sizes[0] == work_sizes[0]);
  CHECK(handler.work_sizes[1] == work_sizes[1]);

  std::cout << "Test Size: " << N << std::endl;

  std::vector<cufftDoubleComplex> h_input(N * N);

  for (int i = 1; i <= N * N; i++) {
    h_input[i - 1].x = 2 * i - 1;
    h_input[i - 1].y = 2 * i;
  }

  cufftDoubleComplex *input;
  cudaMallocAsync((void **)&input, sizeof(cufftDoubleComplex) * N * N, stream);
  cudaMemcpyAsync(input, h_input.data(), sizeof(cufftDoubleComplex) * N * N,
                  cudaMemcpyHostToDevice, stream);

  cufftDoubleComplex h_output[N * N];
  cufftDoubleComplex *output;
  cudaMallocAsync((void **)&output, sizeof(cufftDoubleComplex) * N * N, stream);

  int batch_size = 1;
  int direction = CUFFT_FORWARD;

  cudaDeviceSynchronize();

  // print_2D(h_input.data(), N, N);

  handler.execute(input, output, direction);
  handler.execute(output, input, CUFTT_INVERSE);
  cudaDeviceSynchronize();
  CHECK(handler.result == CUFFT_SUCCESS);

  cudaMemcpyAsync(h_output, input, sizeof(cufftDoubleComplex) * N * N,
                  cudaMemcpyDeviceToHost, stream);

  cudaDeviceSynchronize();

  // print_2D(h_output, N, N);

  // for (int i = 0; i < N * N; i++) {
  //   CHECK(h_output[i].x == doctest::Approx(h_truth[i].x).epsilon(2e-5));
  //   CHECK(h_output[i].y == doctest::Approx(h_truth[i].y).epsilon(2e-5));
  // }

  for (int i = 0; i < N; i++) {
    for(int j = 0; j < N; j++){
      std::cout << i << std::endl;
      std::cout << h_output[i*N+j].x << " " << h_output[i*N+j].y << std::endl;
      std::cout << h_input[i*N+j].x << " " << h_input[i*N+j].y << std::endl;
      std::cout << "-------------------\n";

      // int step = N / num_devices;
      // int block_col_idx = j / step;
      // int local_raveled = i * step + j % step;
      // int global_raveled = block_col_idx * step * N + local_raveled;
      // int new_i = global_raveled / N;
      // int new_j = global_raveled % N;
      //
      // CHECK(h_output[new_i*N+new_j].x == doctest::Approx(h_truth[i*N+j].x).epsilon(2e-5));
      // CHECK(h_output[new_i*N+new_j].y == doctest::Approx(h_truth[i*N+j].y).epsilon(2e-5));
    }
  }


  // free memory
  cudaFree(input);
  cudaFree(output);
  cudaStreamDestroy(stream);
}
