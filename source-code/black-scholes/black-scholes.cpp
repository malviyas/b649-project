/*
 * Copyright (C) 2011 by Justin Holewinski
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 */

#include <iostream>
#include <fstream>
#include <cmath>

#include <sys/time.h>

#include "cuda.h"


typedef float Real;


////////////////////////////////////////////////////////////////////////////////
// Helper function, returning uniformly distributed
// random float in [low, high] range
////////////////////////////////////////////////////////////////////////////////
float RandFloat(float low, float high)
{
    float t = (float)rand() / (float)RAND_MAX;
    return (1.0f - t) * low + t * high;
}

////////////////////////////////////////////////////////////////////////////////
// Data configuration
////////////////////////////////////////////////////////////////////////////////
const int OPT_N = 4000000;
const int  NUM_ITERATIONS = 512;


const int          OPT_SZ = OPT_N * sizeof(float);
const float      RISKFREE = 0.02f;
const float    VOLATILITY = 0.30f;


//==--- Utility Functions --------------------------------------------------== //
const char * statusToString(CUresult error)
{
  switch (error) {
    case CUDA_SUCCESS: return "No errors";
    case CUDA_ERROR_INVALID_VALUE: return "Invalid value";
    case CUDA_ERROR_OUT_OF_MEMORY: return "Out of memory";
    case CUDA_ERROR_NOT_INITIALIZED: return "Driver not initialized";
    case CUDA_ERROR_DEINITIALIZED: return "Driver deinitialized";

    case CUDA_ERROR_NO_DEVICE: return "No CUDA-capable device available";
    case CUDA_ERROR_INVALID_DEVICE: return "Invalid device";

    case CUDA_ERROR_INVALID_IMAGE: return "Invalid kernel image";
    case CUDA_ERROR_INVALID_CONTEXT: return "Invalid context";
    case CUDA_ERROR_CONTEXT_ALREADY_CURRENT: return "Context already current";
    case CUDA_ERROR_MAP_FAILED: return "Map failed";
    case CUDA_ERROR_UNMAP_FAILED: return "Unmap failed";
    case CUDA_ERROR_ARRAY_IS_MAPPED: return "Array is mapped";
    case CUDA_ERROR_ALREADY_MAPPED: return "Already mapped";
    case CUDA_ERROR_NO_BINARY_FOR_GPU: return "No binary for GPU";
    case CUDA_ERROR_ALREADY_ACQUIRED: return "Already acquired";
    case CUDA_ERROR_NOT_MAPPED: return "Not mapped";

    case CUDA_ERROR_INVALID_SOURCE: return "Invalid source";
    case CUDA_ERROR_FILE_NOT_FOUND: return "File not found";

    case CUDA_ERROR_INVALID_HANDLE: return "Invalid handle";

    case CUDA_ERROR_NOT_FOUND: return "Not found";

    case CUDA_ERROR_NOT_READY: return "CUDA not ready";

    case CUDA_ERROR_LAUNCH_FAILED: return "Launch failed";
    case CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES: return "Launch exceeded resources";
    case CUDA_ERROR_LAUNCH_TIMEOUT: return "Launch exceeded timeout";
    case CUDA_ERROR_LAUNCH_INCOMPATIBLE_TEXTURING: return "Launch with incompatible texturing";

    case CUDA_ERROR_UNKNOWN: return "Unknown error";
    default: return "Unknown error ID";
  }
}

void checkSuccess(CUresult    status,
                  const char *func,
                  const char *errorBuffer = 0)
{
  if (status != CUDA_SUCCESS) {
    if (errorBuffer != 0) {
      std::cerr << "ERROR LOG:" << std::endl
                << errorBuffer << std::endl;
    }

    std::cerr << "ERROR: Could not execute '" << func << "', error ("
              << status << ") " << statusToString(status) << std::endl;
    exit(1);
  }
}

double getTimeStamp()
{
  struct timezone Tzp;
  struct timeval  Tp;
  int             stat;
  stat = gettimeofday (&Tp, &Tzp);
  if (stat != 0)
    std::cerr << "Error return from gettimeofday: " << stat << "\n";
  return (Tp.tv_sec + Tp.tv_usec * 1.0e-6);
}





///////////////////////////////////////////////////////////////////////////////
// Polynomial approximation of cumulative normal distribution function
///////////////////////////////////////////////////////////////////////////////





//==--- Entry Point --------------------------------------------------------== //

int main(int argc,
         char** argv) {

 
  CUcontext  context;
  CUdevice   device;
  CUmodule   module;
  CUresult   status;
  CUfunction function;
  
  const int kLogSize = 1024;
  char      logBuffer[kLogSize];

  int blockSizeX        = 480;
  int blockSizeMultiple = 128;
  //int problemSize       = blockSizeX * blockSizeMultiple;
  
  
  // Initialize CUDA
  std::cout << "Initializing CUDA\n";
  checkSuccess(cuInit(0), "cuInit");
  std::cout << "Selecting first compute device\n";
  checkSuccess(cuDeviceGet(&device, 0), "cuDeviceGet");
  std::cout << "Creating CUDA context\n";
  checkSuccess(cuCtxCreate(&context, 0, device), "cuCtxCreate");

  // Read the PTX kernel from disk
  std::ifstream kernelFile("black-scholes.kernel.ptx");
  if (!kernelFile.is_open()) {
    std::cerr << "Failed to open black-scholes.kernel.ptx\n";
    return 1;
  }

  // Load entire kernel into a string
  std::string source(std::istreambuf_iterator<char>(kernelFile),
                     (std::istreambuf_iterator<char>()));

  // Configure JIT options
  CUjit_option jitOptions[] = { CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES,
                                CU_JIT_ERROR_LOG_BUFFER };
  void* jitOptionValues[]   = { reinterpret_cast<void*>(kLogSize), logBuffer };

  // Load the kernel onto the device
  status = cuModuleLoadDataEx(&module, source.c_str(),
                              sizeof(jitOptions)/sizeof(jitOptions[0]),
                              jitOptions, jitOptionValues);
  checkSuccess(status, "cuModuleLoadDataEx", logBuffer);

  status = cuModuleGetFunction(&function, module, "BlackScholesGPU");
  checkSuccess(status, "cuModuleGetFunction");


  // Print some diagnostics about the kernel compilation
  int numRegisters;
  cuFuncGetAttribute(&numRegisters, CU_FUNC_ATTRIBUTE_NUM_REGS, function);
  std::cout << "Register Usage:  " << numRegisters << "\n";
  

  // Setup buffers




  double
  delta, ref, sum_delta, sum_ref, max_delta, L1norm, gpuTime;
  //problemSize = OPT_N;

  Real* h_CallResultCPU = new Real[OPT_N];
  Real* h_PutResultCPU = new Real[OPT_N];

  Real* h_CallResultGPU = new Real[OPT_N];
  Real* h_PutResultGPU = new Real[OPT_N];

  Real* h_StockPrice = new Real[OPT_N];
  Real* h_OptionStrike = new Real[OPT_N];
  Real* h_OptionYears  = new Real[OPT_N];

  std::cout << "Problem Size:  " << OPT_N << "\n";
  
  CUdeviceptr d_CallResult;
  CUdeviceptr d_PutResult;

  CUdeviceptr d_StockPrice;
  CUdeviceptr d_OptionStrike;
  CUdeviceptr d_OptionYears;


  status = cuMemAlloc(&d_CallResult, OPT_N * sizeof(Real));
  checkSuccess(status, "cuMemAlloc");
  status = cuMemAlloc(&d_PutResult, OPT_N * sizeof(Real));
  checkSuccess(status, "cuMemAlloc");

  status = cuMemAlloc(&d_StockPrice, OPT_N * sizeof(Real));
  checkSuccess(status, "cuMemAlloc");
  status = cuMemAlloc(&d_OptionStrike, OPT_N * sizeof(Real));
  checkSuccess(status, "cuMemAlloc");
  status = cuMemAlloc(&d_OptionYears, OPT_N * sizeof(Real));
  checkSuccess(status, "cuMemAlloc");

    for (i = 0; i < OPT_N; i++)
    {
        h_CallResultCPU[i] = 0.0f;
        h_PutResultCPU[i]  = -1.0f;
        h_StockPrice[i]    = RandFloat(5.0f, 30.0f);
        h_OptionStrike[i]  = RandFloat(1.0f, 100.0f);
        h_OptionYears[i]   = RandFloat(0.25f, 10.0f);
    }

  // Copy buffers to device
  status = cuMemcpyHtoD(d_StockPrice, h_StockPrice, OPT_N * sizeof(Real));
  checkSuccess(status, "cuMemcpyHtoD");
  status = cuMemcpyHtoD(d_OptionStrike, h_OptionStrike, OPT_N * sizeof(Real));
  checkSuccess(status, "cuMemcpyHtoD");
  status = cuMemcpyHtoD(d_OptionYears, h_OptionYears, OPT_N * sizeof(Real));
  checkSuccess(status, "cuMemcpyHtoD");


  // Setup block shape
  status = cuFuncSetBlockShape(function, blockSizeX, 1, 1);
  checkSuccess(status, "cuFuncSetBlockShape");
/*
 BlackScholesGPU<<<480, 128>>>(
            d_CallResult,
            d_PutResult,
            d_StockPrice,
            d_OptionStrike,
            d_OptionYears,
            RISKFREE,
            VOLATILITY,
            OPT_N
        );*/
  // Bind kernel paramters
  status = cuParamSetv(function, 0, &d_PutResult, sizeof(CUdeviceptr));
  checkSuccess(status, "cuParamSetv");
  status = cuParamSetv(function, sizeof(CUdeviceptr), &d_PutResult, sizeof(CUdeviceptr));
  checkSuccess(status, "cuParamSetv");
  status = cuParamSetv(function, 2*sizeof(CUdeviceptr), &d_StockPrice, sizeof(CUdeviceptr));
  checkSuccess(status, "cuParamSetv");

  status = cuParamSetv(function, 3*sizeof(CUdeviceptr), &d_OptionStrike, sizeof(CUdeviceptr));
  checkSuccess(status, "cuParamSetv");
  status = cuParamSetv(function, 4*sizeof(CUdeviceptr), &d_OptionYears, sizeof(CUdeviceptr));
  checkSuccess(status, "cuParamSetv");
  status = cuParamSetv(function, 5*sizeof(CUdeviceptr), RISKFREE, sizeof(float));
  checkSuccess(status, "cuParamSetv");

  status = cuParamSetv(function, 5*sizeof(CUdeviceptr)+sizeof(float), VOLATILITY, sizeof(float));
  checkSuccess(status, "cuParamSetv");
  status = cuParamSetv(function, 5*sizeof(CUdeviceptr)+sizeof(float)+sizeof(int), OPT_N, sizeof(int));

  status = cuParamSetSize(function, 5*sizeof(CUdeviceptr)+sizeof(float)+sizeof(int));
  checkSuccess(status, "cuParamSetSize");

  // Launch the kernel
  double deviceStart = getTimeStamp();
  //std::cout << "before:  " << hostX[0] << "s\n"; 
  status = cuLaunchGrid(function, blockSizeMultiple, 1);
  checkSuccess(status, "cuLaunchGrid");
  cuCtxSynchronize();

  double deviceEnd = getTimeStamp();
  

  // Copy results back to the host
  status = cuMemcpyDtoH(h_CallResultGPU, d_CallResult, OPT_N * sizeof(Real));
  checkSuccess(status, "cuMemoryDtoH");
  status = cuMemcpyDtoH(h_PutResultGPU, d_PutResult, OPT_N * sizeof(Real));
  checkSuccess(status, "cuMemoryDtoH");

  //std::cout << "before:  " << hostX[0] << "s\n"; 
  // Compute the reference solution
  //double hostStart = getTimeStamp();
/*
  for (int i = 0; i < problemSize; ++i) {
    refC[i] = hostA[i] + hostB[i];
  }

  double hostEnd = getTimeStamp();


  // Compare the results
  int numWrong = 0;
  
  for (int i = 0; i < problemSize; ++i) {
    if (std::abs(refC[i] - cmpC[i]) > (Real)1e-5) {
      numWrong++;
    }
  }

  if (numWrong == 0) {
    std::cout << "Host reference comparison test PASSED\n";
  }
  else {
    std::cout << "Host reference comparison test FAILED\n";
  }

  std::cout << "Device Time:  " << (deviceEnd - deviceStart) << "s\n";
  std::cout << "Host Time:    " << (hostEnd - hostStart) << "s\n";
  */
  return 0;
}

