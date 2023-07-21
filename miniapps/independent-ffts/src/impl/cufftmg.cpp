#include "cufftmg.hpp"
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime_api.h>
#include <cufft.h>
#include <cufftXt.h>
#include <driver_types.h>
#include <iomanip>
#include <iostream>
#include <stdexcept>

FFTHandler::FFTHandler() {}

FFTHandler::~FFTHandler() {
  // Destroy the plan if it exists
  if (is_initialized) {
    this->result = cufftDestroy(this->plan);
    if (this->result != CUFFT_SUCCESS) {
      fprintf(stderr, "CUFFT error: Plan destruction failed \n");
      fprintf(stderr, "CUFFT error: %d \n", result);
      exit(-1);
    }
  }
}

void FFTHandler::configure(int *device_ids, int size, int num_devices,
                           uint64_t *streams, uint64_t *work_sizes) {

  // Initialize an empty plan
  cudaSetDevice(device_ids[0]);
  this->result = cufftCreate(&this->plan);
  if (this->result != CUFFT_SUCCESS) {
    fprintf(stderr, "CUFFT error: Plan creation failed \n");
    fprintf(stderr, "CUFFT error: %d \n", result);
    exit(-1);
  }

  this->is_initialized = true;
  for (int i = 0; i < num_devices; i++) {
    this->device_ids[i] = device_ids[i];
    this->work_sizes[i] = work_sizes[i];
    this->streams[i] = streams[i];
  }

  this->num_devices = num_devices;
  this->size = size;

  fprintf(stderr, "CUFFT: Configuring plan with %d devices \n", num_devices);
  fprintf(stderr, "CUFFT: Configuring plan with size %d \n", size);
  fprintf(stderr, "CUFFT: Configuring plan with gpus: \n");
  for (int i = 0; i < num_devices; i++) {
    fprintf(stderr, "\tCUDA Device: %d \n", device_ids[i]);
  }

  // Configure the plan
  if (num_devices > 1) {
    this->result = cufftXtSetGPUs(this->plan, num_devices, device_ids);
    if (this->result != CUFFT_SUCCESS) {
      fprintf(stderr, "CUFFT error: Plan GPU association failed \n");
      fprintf(stderr, "CUFFT error: %d \n", result);
      exit(-1);
    }
  }

  // Get the work area size
  this->result =
      cufftGetSize2d(this->plan, size, size, CUFFT_Z2Z, this->work_sizes);
  if (this->result != CUFFT_SUCCESS) {
    fprintf(stderr, "CUFFT error: Plan work area size failed \n");
    fprintf(stderr, "CUFFT error: %d \n", result);
    exit(-1);
  }

  // TODO: Learn how to associate a multigpu plan with streams
  result = cufftSetStream(plan, (cudaStream_t)streams[0]);
  if (result != CUFFT_SUCCESS) {
    fprintf(stderr, "CUFFT error: Plan stream association failed");
    fprintf(stderr, "CUFFT error: %d \n", result);
    exit(-1);
  }

  // Initialize the plan
  // This automatically allocates a workspace
  // NOTE: This memory is inivsibile to CuPy.
  this->result =
      cufftMakePlan2d(this->plan, size, size, CUFFT_Z2Z, this->work_sizes);
  if (this->result != CUFFT_SUCCESS) {
    fprintf(stderr, "CUFFT error: Plan creation failed \n");
    fprintf(stderr, "CUFFT error: %d \n", result);
    throw std::runtime_error("Plan Creation Failed");
    exit(-1);
  }
}

void FFTHandler::set_workspace(uint64_t *workspace) {

  for (int i = 0; i < this->num_devices; i++) {
    this->workspace[i] = reinterpret_cast<cufftDoubleComplex *>(workspace[i]);
  }

  this->result = cufftXtSetWorkArea(this->plan,
                                    reinterpret_cast<void **>(this->workspace));
  if (this->result != CUFFT_SUCCESS) {
    fprintf(stderr, "CUFFT error: Setting workspace failed! \n");
    fprintf(stderr, "CUFFT error: %d \n", result);
    exit(-1);
  }
}

void FFTHandler::execute(void *input, void *output, int direction) {
  // Execute the plan
  if (this->num_devices == 1) {
    this->result = cufftXtExec(this->plan, input, output, direction);
    if (this->result != CUFFT_SUCCESS) {
      fprintf(stderr, "CUFFT error: Plan execution failed\n");
      fprintf(stderr, "CUFFT error: %d \n", this->result);
      exit(-1);
    }
  } else {
    // Assume the data is on the host and we need to copy it to the device
    cudaLibXtDesc *d_f, *d_d_f, *d_out;

    this->result = cufftXtMalloc(this->plan, &d_f, CUFFT_XT_FORMAT_INPLACE);
    if (this->result != CUFFT_SUCCESS) {
      fprintf(stderr, "CUFFT error: Plan malloc failed\n");
      fprintf(stderr, "CUFFT error: %d \n", this->result);
      exit(-1);
    }

    this->result =
        cufftXtMemcpy(this->plan, d_f, input, CUFFT_COPY_HOST_TO_DEVICE);
    if (this->result != CUFFT_SUCCESS) {
      fprintf(stderr, "CUFFT error: Plan memcpy (H2D) failed\n");
      fprintf(stderr, "CUFFT error: %d \n", this->result);
      exit(-1);
    }

    /*
    std::cout << "Version: " << d_f->version << std::endl;
    std::cout << "Version Internal: " << d_f->descriptor->version << std::endl;
    std::cout << "Size: " << std::endl;
    for (int i = 0; i < this->num_devices; i++) {
      std::cout << d_f->descriptor->size[i] << std::endl;
    }
    std::cout << "GPUs: " << std::endl;
    for (int i = 0; i < this->num_devices; i++) {
      std::cout << d_f->descriptor->GPUs[i] << std::endl;
    }
    std::cout << "cudaXtState" << d_f->descriptor->cudaXtState << std::endl;
    std::cout << "libdescriptor" << d_f->libDescriptor << std::endl;
    std::cout << "ngpus " << d_f->descriptor->nGPUs << std::endl;
    std::cout << "libFormat" << d_f->library << std::endl;
    */

    // All CUFFTMG support is inplace, input == output
    this->result = cufftXtExecDescriptorZ2Z(this->plan, d_f, d_f, direction);
    if (this->result != CUFFT_SUCCESS) {
      fprintf(stderr, "CUFFT error: Plan execution failed\n");
      fprintf(stderr, "CUFFT error: %d \n", this->result);
      exit(-1);
    }

    this->result =
        cufftXtMemcpy(this->plan, output, d_f, CUFFT_COPY_DEVICE_TO_HOST);
    if (this->result != CUFFT_SUCCESS) {
      fprintf(stderr, "CUFFT error: Plan memcpy (D2H) failed\n");
      fprintf(stderr, "CUFFT error: %d \n", this->result);
      exit(-1);
    }

    this->result = cufftXtFree(d_f);
    if (this->result != CUFFT_SUCCESS) {
      fprintf(stderr, "CUFFT error: Plan free failed\n");
      fprintf(stderr, "CUFFT error: %d \n", this->result);
      exit(-1);
    }
  }
}

void FFTHandler::execute(void **input, void **output, int direction) {
  if (this->num_devices == 1) {
    FFTHandler::execute((void *)input[0], (void *)output[0], direction);
  } else {

    fprintf(stderr, "Running multidevice FFT!\n");

    // Setup the descriptors
    cudaXtDesc *input_descriptor;
    cudaXtDesc *output_descriptor;
    input_descriptor = (cudaXtDesc *)calloc(1, sizeof(cudaXtDesc));
    output_descriptor = (cudaXtDesc *)calloc(1, sizeof(cudaXtDesc));

    // Initialize descriptor
    input_descriptor->nGPUs = num_devices;
    output_descriptor->nGPUs = num_devices;

    // Initialize memory space for each gpu
    for (int i = 0; i < num_devices; i++) {
      // std::cout << "Device ID: " << this->device_ids[i] << std::endl;
      input_descriptor->GPUs[i] = this->device_ids[i];
      output_descriptor->GPUs[i] = this->device_ids[i];

      input_descriptor->data[i] = input[i];
      output_descriptor->data[i] = output[i];

      const int local_partition = this->size / num_devices;
      input_descriptor->size[i] =
          local_partition * this->size * sizeof(cufftDoubleComplex);

      output_descriptor->size[i] = input_descriptor->size[i];
    }

    cudaLibXtDesc *input_outer_descriptor;
    cudaLibXtDesc *output_outer_descriptor;
    input_outer_descriptor = (cudaLibXtDesc *)calloc(1, sizeof(cudaLibXtDesc));
    output_outer_descriptor = (cudaLibXtDesc *)calloc(1, sizeof(cudaLibXtDesc));

    input_outer_descriptor->descriptor = input_descriptor;
    output_outer_descriptor->descriptor = output_descriptor;

    if (direction == CUFFT_INVERSE) {
      input_outer_descriptor->subFormat = CUFFT_XT_FORMAT_INPLACE_SHUFFLED;
    } else {
      input_outer_descriptor->subFormat = CUFFT_XT_FORMAT_INPLACE;
    }

    output_outer_descriptor->subFormat = CUFFT_XT_FORMAT_INPLACE;
    // lib_descriptor->version = version;

    fprintf(stderr, "Created descriptors\n");
    fprintf(stderr, "Input[0]: %p\n",
            input_outer_descriptor->descriptor->data[0]);
    fprintf(stderr, "Input[1]: %p\n",
            input_outer_descriptor->descriptor->data[1]);
    fprintf(stderr, "Direction: %d\n", direction);

    // All CUFFTMG support is inplace, input == output
    this->result = cufftXtExecDescriptorZ2Z(this->plan, input_outer_descriptor,
                                            input_outer_descriptor, direction);

    cudaSetDevice(this->device_ids[0]);
    cudaEvent_t fft_event;
    cudaEventCreate(&fft_event);
    cudaEventRecord(fft_event, (cudaStream_t)this->streams[0]);

    for (int i = 0; i < num_devices; i++) {
      cudaSetDevice(this->device_ids[i]);
      cudaStreamWaitEvent((cudaStream_t)this->streams[i], fft_event, 0);
    }
    cudaSetDevice(this->device_ids[0]);

    if (this->result != CUFFT_SUCCESS) {
      fprintf(stderr, "CUFFT error: Plan execution failed\n");
      fprintf(stderr, "CUFFT error: %d \n", this->result);
      free(input_descriptor);
      free(input_outer_descriptor);
      free(output_descriptor);
      free(output_outer_descriptor);
      exit(-1);
    }

    // input_outer_descriptor->subFormat = CUFFT_XT_FORMAT_INPLACE_SHUFFLED;

    // Copy input to output on each device
    // this->result =
    //    cufftXtMemcpy(this->plan, output_outer_descriptor,
    //                  input_outer_descriptor, CUFFT_COPY_DEVICE_TO_DEVICE);

    // if (this->result != CUFFT_SUCCESS) {
    //   fprintf(stderr, "CUFFT error: Plan memcpy (D2D) failed\n");
    //   fprintf(stderr, "CUFFT error: %d \n", this->result);
    //   free(input_descriptor);
    //   free(input_outer_descriptor);
    //   free(output_descriptor);
    //   free(output_outer_descriptor);
    //   exit(-1);
    // }

    free(input_descriptor);
    free(input_outer_descriptor);
    free(output_descriptor);
    free(output_outer_descriptor);
  }
}
