#include <cuda_runtime_api.h>
#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
// #include <cudalibxt.h>
#include <cufftXt.h>
// #include <driver_types.h>
#include <algorithm>
#include <cuda_runtime.h>
#include <cufft.h>
#include <cufftmg.hpp>
#include <doctest.h>
#include <iomanip>
#include <iostream>
#include <list>
#include <string>
#include <tuple>

void print_1D(cufftComplex *array, int size) {
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

void print_2D(cufftComplex *array, int rows, int cols) {
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

TEST_CASE("FFT Handler: 2 GPUs - Init on Host") {
  cufftmg::FFTHandler handler = cufftmg::FFTHandler();
  int num_devices = 4;
  int device_ids[num_devices] = {0, 1, 2, 3};
  int N = 40;

  uintptr_t streams[num_devices];

  size_t work_sizes[num_devices] = {0, 0, 0, 0};

  std::cout << "Test Size: " << N << std::endl;

  // cufftComplex h_input[] = {{1, 2},   {3, 4},   {5, 6},   {7, 8},
  //                           {9, 10},  {11, 12}, {13, 14}, {15, 16},
  //                           {17, 18}, {19, 20}, {21, 22}, {23, 24},
  //                           {25, 26}, {27, 28}, {29, 30}, {31, 32}};

  cufftComplex *h_input = (cufftComplex *)calloc(N * N, sizeof(cufftComplex));

  cudaSetDevice(device_ids[0]);

  handler.configure(device_ids, N, num_devices, streams, work_sizes);

  CHECK(handler.num_devices == num_devices);
  CHECK(handler.size == N);
  CHECK(handler.device_ids[0] == device_ids[0]);
  CHECK(handler.device_ids[1] == device_ids[1]);
  CHECK(handler.streams[0] == streams[0]);
  CHECK(handler.streams[1] == streams[1]);
  CHECK(handler.work_sizes[0] == work_sizes[0]);
  CHECK(handler.work_sizes[1] == work_sizes[1]);

  cufftComplex *h_output = (cufftComplex *)calloc(N * N, sizeof(cufftComplex));

  int direction = CUFFT_FORWARD;

  cudaDeviceSynchronize();
  // print_2D(h_input, N, N);

  handler.execute(h_input, h_output, direction);
  cudaDeviceSynchronize();
  CHECK(handler.result == CUFFT_SUCCESS);

  cudaSetDevice(device_ids[0]);

  std::cout << "OUTPUT: " << std::endl;
  // print_2D(h_output, N, N);

  // cufftComplex h_expected[N * N] = {{256, 272}, {-32, 0}, {-16, -16}, {0,
  // -32},
  //                                   {-128, 0},  {0, 0},   {0, 0},     {0, 0},
  //                                   {-64, -64}, {0, 0},   {0, 0},     {0, 0},
  //                                   {0, -128},  {0, 0},   {0, 0},     {0,
  //                                   0}};

  // std::cout << "EXPECTED: " << std::endl;
  // print_2D(h_expected, N, N);

  // for (int i = 0; i < N * N; i++) {
  //   CHECK(h_output[i].x == doctest::Approx(h_expected[i].x));
  //   CHECK(h_output[i].y == doctest::Approx(h_expected[i].y));
  // }

  // free memory
  free(h_input);
  free(h_output);
}

TEST_CASE("FFT Handler: 2 GPUs - Init on Device") {
  cufftmg::FFTHandler handler = cufftmg::FFTHandler();
  int num_devices = 4;
  int device_ids[num_devices] = {0, 1, 2, 3};
  int N = 40;

  uintptr_t streams[num_devices];

  size_t work_sizes[num_devices] = {0, 0, 0, 0};

  std::cout << "Test Size: " << N << std::endl;

  // cufftComplex h_input[] = {{1, 2},   {3, 4},   {5, 6},   {7, 8},
  //                           {9, 10},  {11, 12}, {13, 14}, {15, 16},
  //                           {17, 18}, {19, 20}, {21, 22}, {23, 24},
  //                           {25, 26}, {27, 28}, {29, 30}, {31, 32}};

  cufftComplex *h_input = (cufftComplex *)calloc(N * N, sizeof(cufftComplex));

  // Initialize memory space for each gpu
  cufftComplex *data[num_devices];
  for (int i = 0; i < num_devices; i++) {
    cudaSetDevice(device_ids[i]);
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    streams[i] = (uintptr_t)stream;
    const int local_partition = N / num_devices;
    cudaMalloc((cufftComplex **)&data[i],
               sizeof(cufftComplex) * local_partition * N);

    cudaMemcpy(data[i], h_input + i * local_partition * N,
               sizeof(cufftComplex) * local_partition * N,
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
  CHECK(handler.work_sizes[0] == work_sizes[0]);
  CHECK(handler.work_sizes[1] == work_sizes[1]);

  cufftComplex *h_output = (cufftComplex *)calloc(N * N, sizeof(cufftComplex));

  int direction = CUFFT_FORWARD;

  cudaDeviceSynchronize();
  // print_2D(h_input, N, N);

  handler.execute(data, data, direction);
  cudaDeviceSynchronize();
  CHECK(handler.result == CUFFT_SUCCESS);

  // copy results back to host
  for (int i = 0; i < num_devices; i++) {
    cudaSetDevice(device_ids[i]);
    const int local_partition = N / num_devices;
    cudaMemcpy(h_output + i * local_partition * N, data[i],
               sizeof(cufftComplex) * local_partition * N,
               cudaMemcpyDeviceToHost);
  }
  cudaSetDevice(device_ids[0]);

  std::cout << "OUTPUT: " << std::endl;
  // print_2D(h_output, N, N);

  // cufftComplex h_expected[N * N] = {{256, 272}, {-32, 0}, {-16, -16}, {0,
  // -32},
  //                                   {-128, 0},  {0, 0},   {0, 0},     {0, 0},
  //                                   {-64, -64}, {0, 0},   {0, 0},     {0, 0},
  //                                   {0, -128},  {0, 0},   {0, 0},     {0,
  //                                   0}};

  // std::cout << "EXPECTED: " << std::endl;
  // print_2D(h_expected, N, N);

  // for (int i = 0; i < N * N; i++) {
  //   CHECK(h_output[i].x == doctest::Approx(h_expected[i].x));
  //   CHECK(h_output[i].y == doctest::Approx(h_expected[i].y));
  // }

  // free memory
  for (int i = 0; i < num_devices; i++) {
    cudaSetDevice(device_ids[i]);
    cudaStreamDestroy((cudaStream_t)streams[i]);
    cudaFree(data[i]);
  }
  cudaSetDevice(device_ids[0]);
  free(h_input);
  free(h_output);
}
