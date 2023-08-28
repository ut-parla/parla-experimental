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

/*
TEST_CASE("FFT Handler: 2 GPUs - Init on Host") {
  FFTHandler handler = FFTHandler();
  const int num_devices = 2;
  int device_ids[num_devices] = {0, 1};

  for(int i = 0; i < num_devices; i++){
    std::cout << "OUTER DEVICE ID: " << device_ids[i] << std::endl;
  }
  const int N = 40;

  std::vector<cufftDoubleComplex> h_truth;
  read_binary(h_truth, "../data/40_fft.bin");

  uintptr_t streams[num_devices] = {0};

  size_t work_sizes[num_devices] = {0};

  std::cout << "Test Size: " << N << std::endl;

  cufftDoubleComplex *h_input =
      (cufftDoubleComplex *)calloc(N * N, sizeof(cufftDoubleComplex));

  for (int i = 1; i <= N * N; i++) {
    h_input[i - 1].x = 2 * i - 1;
    h_input[i - 1].y = 2 * i;
  }

  cudaSetDevice(device_ids[0]);

  handler.configure(device_ids, N, num_devices, streams, work_sizes);

  CHECK(handler.num_devices == num_devices);
  CHECK(handler.size == N);
  CHECK(handler.device_ids[0] == device_ids[0]);
  CHECK(handler.device_ids[1] == device_ids[1]);
  CHECK(handler.streams[0] == streams[0]);
  CHECK(handler.streams[1] == streams[1]);

  cufftDoubleComplex *h_output =
      (cufftDoubleComplex *)calloc(N * N, sizeof(cufftDoubleComplex));

  int direction = CUFFT_FORWARD;

  cudaDeviceSynchronize();
  //print_2D(h_input, N, N);

  handler.execute(h_input, h_output, direction);
  cudaDeviceSynchronize();
  CHECK(handler.result == CUFFT_SUCCESS);

  cudaSetDevice(device_ids[0]);

  for (int i = 0; i < N * N; i++) {

    // std::cout << i << std::endl;
    // std::cout << h_output[i].x << " " << h_output[i].y << std::endl;
    // std::cout << h_truth[i].x << " " << h_truth[i].y << std::endl;
    // std::cout << "-------------------\n";

    CHECK(h_output[i].x == doctest::Approx(h_truth[i].x).epsilon(2e-5));
    CHECK(h_output[i].y == doctest::Approx(h_truth[i].y).epsilon(2e-5));
  }

  // free memory
  free(h_input);
  free(h_output);
}
*/

TEST_CASE("FFT Handler: 2 GPUs - Init on Device") {
  FFTHandler handler = FFTHandler();
  const int num_devices = 2;
  const int N = 40;

  int device_ids[num_devices] = {0, 1};

  std::vector<cufftDoubleComplex> h_truth;
  read_binary(h_truth, "../data/40_fft.bin");

  uintptr_t streams[num_devices] = {0};

  size_t work_sizes[num_devices] = {0};

  std::cout << "Test Size: " << N << std::endl;

  cufftDoubleComplex *h_input =
      (cufftDoubleComplex *)calloc(N * N, sizeof(cufftDoubleComplex));

  for (int i = 1; i <= N * N; i++) {
    h_input[i - 1].x = 2 * i - 1;
    h_input[i - 1].y = 2 * i;
  }

  // Initialize memory space for each gpu
  cufftDoubleComplex *d_input[num_devices];
  cufftDoubleComplex *d_output[num_devices];

  for (int i = 0; i < num_devices; i++) {
    cudaSetDevice(device_ids[i]);
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    streams[i] = (uintptr_t)stream;
    const int local_partition = N / num_devices;
    cudaMalloc((cufftDoubleComplex **)&d_input[i],
               sizeof(cufftDoubleComplex) * local_partition * N);

    cudaMalloc((cufftDoubleComplex **)&d_output[i],
               sizeof(cufftDoubleComplex) * local_partition * N);

    cudaMemcpy(d_input[i], h_input + i * local_partition * N,
               sizeof(cufftDoubleComplex) * local_partition * N,
               cudaMemcpyHostToDevice);
  }
  cudaSetDevice(device_ids[0]);

  handler.configure(device_ids, N, num_devices, streams, work_sizes);

  CHECK(handler.num_devices == num_devices);
  CHECK(handler.size == N);
  CHECK(handler.device_ids[0] == device_ids[0]);
  CHECK(handler.device_ids[1] == device_ids[1]);
  CHECK(handler.streams[0] == streams[0]);
  CHECK(handler.streams[1] == streams[1]);

  cufftDoubleComplex *h_output =
      (cufftDoubleComplex *)calloc(N * N, sizeof(cufftDoubleComplex));

  const int forward_direction = CUFFT_FORWARD;
  const int inverse_direction = CUFFT_INVERSE;

  std::cout << "Forward Direction:  " << forward_direction << std::endl;
  std::cout << "Backward Direction: " << inverse_direction << std::endl;

  cudaDeviceSynchronize();
  // print_2D(h_input, N, N);

  handler.execute((void**) d_input, (void**) d_input, forward_direction);
  CHECK(handler.result == CUFFT_SUCCESS);
  cudaDeviceSynchronize();
  // handler.execute((void**) d_input, (void**) d_output, inverse_direction);
  // CHECK(handler.result == CUFFT_SUCCESS);
  // cudaDeviceSynchronize();

  // copy results back to host
  for (int i = 0; i < num_devices; i++) {
    cudaSetDevice(device_ids[i]);
    const int local_partition = N / num_devices;
    cudaMemcpy(h_output + i * local_partition * N, d_input[i],
               sizeof(cufftDoubleComplex) * local_partition * N,
               cudaMemcpyDeviceToHost);
  }
  cudaSetDevice(device_ids[0]);

  for (int i = 0; i < N; i++) {
    for(int j = 0; j < N; j++){
      // std::cout << i << std::endl;
      // std::cout << h_output[i*N+j].x << " " << h_output[i*N+j].y << std::endl;
      // std::cout << h_input[i*N+j].x << " " << h_input[i*N+j].y << std::endl;
      // std::cout << "-------------------\n";

      int step = N / num_devices;
      int block_col_idx = j / step;
      int local_raveled = i * step + j % step;
      int global_raveled = block_col_idx * step * N + local_raveled;
      int new_i = global_raveled / N;
      int new_j = global_raveled % N;

      CHECK(h_output[new_i*N+new_j].x == doctest::Approx(h_truth[i*N+j].x).epsilon(2e-5));
      CHECK(h_output[new_i*N+new_j].y == doctest::Approx(h_truth[i*N+j].y).epsilon(2e-5));
    }
  }

  // free memory
  for (int i = 0; i < num_devices; i++) {
    cudaSetDevice(device_ids[i]);
    cudaStreamDestroy((cudaStream_t)streams[i]);
    cudaFree(d_input[i]);
    cudaFree(d_output[i]);
  }
  cudaSetDevice(device_ids[0]);
  free(h_input);
  free(h_output);
}
