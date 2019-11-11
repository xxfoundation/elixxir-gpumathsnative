/***

Copyright (c) 2018-2019, NVIDIA CORPORATION.  All rights reserved.

Permission is hereby granted, free of charge, to any person obtaining a
copy of this software and associated documentation files (the "Software"),
to deal in the Software without restriction, including without limitation
the rights to use, copy, modify, merge, publish, distribute, sublicense,
and/or sell copies of the Software, and to permit persons to whom the
Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
IN THE SOFTWARE.

***/

#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <cuda.h>
#include <cuda_profiler_api.h>
#include <gmp.h>
#include "cgbn/cgbn.h"
#include "../utility/support.h"
#include "powm_odd_export.h"

#define TRACE

// Stream object and associated data for a stream
// This name could perhaps be better...
struct streamData {
  // This CUDA stream; when performing operations, set the current stream to
  // this one
  cudaStream_t stream;

  // Device buffer uploaded to before execution
  // Contains multiple items
  void *gpuInputs;
  // Device buffer downloaded from after execution
  // Contains multiple items
  void *gpuOutputs;
  // Device data that's the same for all items, uploaded to before execution
  // For instance, can contain the modulus.
  // Note: This currently uses global memory, not constant memory.
  // It isn't a constant buffer.
  void *gpuConstants;
  
  // Host buffer downloaded to after execution
  // This is allocated with pinned memory that can be transferred with the DMA
  void *cpuOutputs;
  void *cpuInputs;
  void *cpuConstants;

  // Size of buffers at max capacity (set at creation time)
  size_t inputsCapacity;
  size_t outputsCapacity;
  size_t constantsCapacity;

  // Size of uploads and downloads for this launch (set at upload time)
  size_t inputsLength;
  size_t outputsLength;
  size_t constantsLength;

  // Number of items that can be held in the buffers associated with this stream
  size_t capacity;
  // Number of items to be processed with this part of the stream
  size_t length;

  // Check for CGBN errors after kernel finishes using this
  cgbn_error_report_t *report;
  // Synchronize this event to wait for host to device transfer before kernel execution
  cudaEvent_t hostToDevice;
  // Synchronize this event to wait for kernel execution to finish before device to host transfer
  cudaEvent_t exec;
  // Synchronize this event to wait for downloading to finish before using results
  cudaEvent_t deviceToHost;
};

// For this example, there are quite a few template parameters that are used to generate the actual code.
// In order to simplify passing many parameters, we use the same approach as the CGBN library, which is to
// create a container class with static constants and then pass the class.

// The CGBN context uses the following three parameters:
//   TBP             - threads per block (zero means to use the blockDim.x)
//   MAX_ROTATION    - must be small power of 2, imperically, 4 works well
//   SHM_LIMIT       - number of bytes of dynamic shared memory available to the kernel
//   CONSTANT_TIME   - require constant time algorithms (currently, constant time algorithms are not available)

// Locally it will also be helpful to have several parameters:
//   TPI             - threads per instance
//   BITS            - number of bits per instance
//   WINDOW_BITS     - number of bits to use for the windowed exponentiation

template<uint32_t tpi, uint32_t bits, uint32_t window_bits>
class powm_params_t {
  public:
  // parameters used by the CGBN context
  static const uint32_t TPB=0;                     // get TPB from blockDim.x  
  static const uint32_t MAX_ROTATION=4;            // good default value
  static const uint32_t SHM_LIMIT=0;               // no shared mem available
  static const bool     CONSTANT_TIME=false;       // constant time implementations aren't available yet
  
  // parameters used locally in the application
  static const uint32_t TPI=tpi;                   // threads per instance
  static const uint32_t BITS=bits;                 // instance size
  static const uint32_t WINDOW_BITS=window_bits;   // window size
};

template<class params>
class cmixPrecomp {
  public:
  static const uint32_t window_bits=params::WINDOW_BITS;  // used a lot, give it an instance variable

  typedef cgbn_mem_t<params::BITS> mem_t;

  typedef struct {
    mem_t x;
    mem_t power;
  } powm_odd_input_t;

  typedef struct {
    // Used to calculate both outputs
    mem_t privateKey;

    // Used to calculate ecrKeys output
    mem_t ecrKeys;
    mem_t key;

    // Used to calculate cypher output
    mem_t publicCypherKey;
    mem_t cypher;
  } elgamal_input_t;

  typedef struct {
    mem_t ecrKeys;
    mem_t cypher;
  } elgamal_output_t;

  typedef struct {
    mem_t g;
    mem_t prime;
  } elgamal_constant_t;

  
  typedef cgbn_context_t<params::TPI, params>   context_t;
  typedef cgbn_env_t<context_t, params::BITS>   env_t;
  typedef typename env_t::cgbn_t                bn_t;
  typedef typename env_t::cgbn_local_t          bn_local_t;

  context_t _context;
  env_t     _env;

  __device__ __forceinline__ cmixPrecomp(cgbn_monitor_t monitor, cgbn_error_report_t *report, int32_t instance) : _context(monitor, report, (uint32_t)instance), _env(_context) {
  }

  // Precondition: x is in montgomery space corresponding to modulus
  // Doesn't convert into or out of montgomery space, as I'm trying to decouple the exponentiations and multiplications from the montgomery conversions
  __device__ __forceinline__ void fixed_window_powm_odd(bn_t &result, const bn_t &x, const bn_t &power, const bn_t &modulus, uint32_t np0) {
    bn_t       t;
    bn_local_t window[1<<window_bits];
    int32_t    index, position, offset;

    // conmpute x^power mod modulus, using the fixed window algorithm
    // requires:  x<modulus,  modulus is odd

    // compute x^0 (in Montgomery space, this is just 2^BITS - modulus)
    cgbn_negate(_env, t, modulus);
    cgbn_store(_env, window+0, t);
    
    // store x into window table
    cgbn_set(_env, result, x);
    cgbn_store(_env, window+1, result);
    cgbn_set(_env, t, result);
    
    // compute x^2, x^3, ... x^(2^window_bits-1), store into window table
    #pragma nounroll
    for(index=2;index<(1<<window_bits);index++) {
      cgbn_mont_mul(_env, result, result, t, modulus, np0);
      cgbn_store(_env, window+index, result);
    }

    // find leading high bit
    position=params::BITS - cgbn_clz(_env, power);

    // break the exponent into chunks, each window_bits in length
    // load the most significant non-zero exponent chunk
    offset=position % window_bits;
    if(offset==0)
      position=position-window_bits;
    else
      position=position-offset;
    index=cgbn_extract_bits_ui32(_env, power, position, window_bits);
    cgbn_load(_env, result, window+index);
    
    // process the remaining exponent chunks
    while(position>0) {
      // square the result window_bits times
      #pragma nounroll
      for(int sqr_count=0;sqr_count<window_bits;sqr_count++)
        cgbn_mont_sqr(_env, result, result, modulus, np0);
      
      // multiply by next exponent chunk
      position=position-window_bits;
      index=cgbn_extract_bits_ui32(_env, power, position, window_bits);
      cgbn_load(_env, t, window+index);
      cgbn_mont_mul(_env, result, result, t, modulus, np0);
    }
    
    // we've processed the exponent now, and at some point it will need to be converted to normal space again
  }
  
  // Precondition: Inputs are not transformed to Montgomery space
  __device__ __forceinline__ void sliding_window_powm_odd(bn_t &result, const bn_t &x, const bn_t &power, const bn_t &modulus) {
    bn_t         t, starts;
    int32_t      index, position, leading;
    uint32_t     mont_inv;
    bn_local_t   odd_powers[1<<window_bits-1];

    // compute x^power mod modulus, using Constant Length Non-Zero windows (CLNZ).
    // requires:  x<modulus,  modulus is odd
        
    // find the leading one in the power
    leading=params::BITS-1-cgbn_clz(_env, power);
    if(leading>=0) {
      // convert x into Montgomery space, store in the odd powers table
      mont_inv=cgbn_bn2mont(_env, result, x, modulus);
      
      // compute t=x^2 mod modulus
      cgbn_mont_sqr(_env, t, result, modulus, mont_inv);
      
      // compute odd powers window table: x^1, x^3, x^5, ...
      cgbn_store(_env, odd_powers, result);
      #pragma nounroll
      for(index=1;index<(1<<window_bits-1);index++) {
        cgbn_mont_mul(_env, result, result, t, modulus, mont_inv);
        cgbn_store(_env, odd_powers+index, result);
      }
  
      // starts contains an array of bits indicating the start of a window
      cgbn_set_ui32(_env, starts, 0);
  
      // organize p as a sequence of odd window indexes
      position=0;
      while(true) {
        if(cgbn_extract_bits_ui32(_env, power, position, 1)==0)
          position++;
        else {
          cgbn_insert_bits_ui32(_env, starts, starts, position, 1, 1);
          if(position+window_bits>leading)
            break;
          position=position+window_bits;
        }
      }
  
      // load first window.  Note, since the window index must be odd, we have to
      // divide it by two before indexing the window table.  Instead, we just don't
      // load the index LSB from power
      index=cgbn_extract_bits_ui32(_env, power, position+1, window_bits-1);
      cgbn_load(_env, result, odd_powers+index);
      position--;
      
      // Process remaining windows 
      while(position>=0) {
        cgbn_mont_sqr(_env, result, result, modulus, mont_inv);
        if(cgbn_extract_bits_ui32(_env, starts, position, 1)==1) {
          // found a window, load the index
          index=cgbn_extract_bits_ui32(_env, power, position+1, window_bits-1);
          cgbn_load(_env, t, odd_powers+index);
          cgbn_mont_mul(_env, result, result, t, modulus, mont_inv);
        }
        position--;
      }
      
      // convert result from Montgomery space
      cgbn_mont2bn(_env, result, result, modulus, mont_inv);
    }
    else {
      // p=0, thus x^p mod modulus=1
      cgbn_set_ui32(_env, result, 1);
    }
  }
};

// kernel implementation using cgbn
// 
// Unfortunately, the kernel must be separate from the cmixPrecomp class
// kernel_powm_odd<params><<<(instance_count+IPB-1)/IPB, TPB>>>(report, gpuInputs, gpuResults, instance_count);
template<class params>
__global__ void kernel_powm_odd(cgbn_error_report_t *report, typename cmixPrecomp<params>::powm_odd_input_t *inputs, cgbn_mem_t<params::BITS> *modulus, cgbn_mem_t<params::BITS> *outputs, size_t count) {
  int32_t instance;

  // decode an instance number from the blockIdx and threadIdx
  instance=(blockIdx.x*blockDim.x + threadIdx.x)/params::TPI;
  if(instance>=count)
    return;

  cmixPrecomp<params>                 po(cgbn_report_monitor, report, instance);
  typename cmixPrecomp<params>::bn_t  r, x, p, m;
  
  // the loads and stores can go in the class, but it seems more natural to have them
  // here and to pass in and out bignums
  cgbn_load(po._env, x, &(inputs[instance].x));
  cgbn_load(po._env, p, &(inputs[instance].power));
  cgbn_load(po._env, m, modulus);
  
  // this can be either fixed_window_powm_odd or sliding_window_powm_odd.
  // when TPI<32, fixed window runs much faster because it is less divergent, so we use it here
  uint32_t np0 = cgbn_bn2mont(po._env, x, x, m);
  po.fixed_window_powm_odd(r, x, p, m, np0);
  cgbn_mont2bn(po._env, r, r, m, np0);
  //   OR
  // po.sliding_window_powm_odd(r, x, p, m);
  
  cgbn_store(po._env, &(outputs[instance]), r);
}

template<class params>
__global__ void kernel_elgamal(cgbn_error_report_t *report, typename cmixPrecomp<params>::elgamal_input_t *inputs, typename cmixPrecomp<params>::elgamal_constant_t *constants, typename cmixPrecomp<params>::elgamal_output_t *outputs, size_t count) {
  int32_t instance;

  // decode an instance number from the blockIdx and threadIdx
  instance=(blockIdx.x*blockDim.x + threadIdx.x)/params::TPI;
  if(instance>=count)
    return;

  cmixPrecomp<params>                 po(cgbn_report_monitor, report, instance);
  typename cmixPrecomp<params>::bn_t privateKey, ecrKeys, key, publicCypherKey, cypher, g, prime, r;

  // Prepare elgamal inputs
  cgbn_load(po._env, privateKey, &(inputs[instance].privateKey));
  cgbn_load(po._env, ecrKeys, &(inputs[instance].ecrKeys));
  cgbn_load(po._env, key, &(inputs[instance].key));
  cgbn_load(po._env, publicCypherKey, &(inputs[instance].publicCypherKey));
  cgbn_load(po._env, cypher, &(inputs[instance].cypher));
  cgbn_load(po._env, g, &(constants->g));
  cgbn_load(po._env, prime, &(constants->prime));

  // Calculate ecrKeys first
  uint32_t np0 = cgbn_bn2mont(po._env, g, g, prime);
  po.fixed_window_powm_odd(r, g, privateKey, prime, np0);
  cgbn_mont_mul(po._env, r, r, key, prime, np0);
  cgbn_mont_mul(po._env, r, r, ecrKeys, prime, np0);
  cgbn_mont2bn(po._env, ecrKeys, r, prime, np0);
  cgbn_store(po._env, &(outputs[instance].ecrKeys), ecrKeys);

  // Calculate publicCypherKey second
  np0 = cgbn_bn2mont(po._env, publicCypherKey, publicCypherKey, prime);
  po.fixed_window_powm_odd(r, publicCypherKey, privateKey, prime, np0);
  cgbn_mont_mul(po._env, r, r, cypher, prime, np0);
  cgbn_mont2bn(po._env, cypher, r, prime, np0);
  cgbn_store(po._env, &(outputs[instance].cypher), cypher);
}

// Run powm kernel
// Enqueues kernel on the stream and returns immediately (non-blocking)
// The results will be placed in the stream's gpu outputs buffer some time after the kernel launch
// Precondition: stream should have had upload called on it
template<class params>
const char* run(streamData *stream, kernel whichToRun) {
#ifdef TRACE
  printf("run (streamData, kernel)\n");
#endif
  const int32_t              TPB=(params::TPB==0) ? 128 : params::TPB;    // default threads per block to 128
  const int32_t              TPI=params::TPI, IPB=TPB/TPI;                // IPB is instances per block

  CUDA_CHECK_RETURN(cudaStreamWaitEvent(stream->stream, stream->hostToDevice, 0));
  // launch kernel with blocks=ceil(instance_count/IPB) and threads=TPB
  // TODO We should be able to launch more than just this kernel.
  //  Organize with enumeration? Is it possible to use templates to make this better?
  typedef cgbn_mem_t<params::BITS> mem_t;

  switch (whichToRun) {
  case KERNEL_POWM_ODD:
    {
      typedef typename cmixPrecomp<params>::powm_odd_input_t input_t;
      kernel_powm_odd<params><<<(stream->length+IPB-1)/IPB, TPB, 0, stream->stream>>>(
        stream->report, 
        (input_t*)stream->gpuInputs, 
        (mem_t*)stream->gpuConstants, 
        (mem_t*)stream->gpuOutputs, 
        stream->length);
    }
    break;
  case KERNEL_ELGAMAL:
    {
      typedef typename cmixPrecomp<params>::elgamal_input_t input_t;
      typedef typename cmixPrecomp<params>::elgamal_output_t output_t;
      typedef typename cmixPrecomp<params>::elgamal_constant_t constant_t;
      kernel_elgamal<params><<<(stream->length+IPB-1)/IPB, TPB, 0, stream->stream>>>(
        stream->report, 
        (input_t*)stream->gpuInputs, 
        (constant_t*)stream->gpuConstants, 
        (output_t*)stream->gpuOutputs, 
        stream->length);
    }
    break;
  case KERNEL_MUL2:
    return strdup("KERNEL_MUL2 unimplemented");
    break;
  default:
    return strdup("Unknown kernel not implemented");
    break;
  }

  CUDA_CHECK_RETURN(cudaEventRecord(stream->exec, stream->stream));

  return NULL;
}

const char* getResults(streamData *stream) {
#ifdef TRACE
  printf("getResults (streamData)\n");
#endif
  // Wait for download to complete
  CUDA_CHECK_RETURN(cudaEventSynchronize(stream->deviceToHost));
  // Not sure if we can check the error report before this (e.g. in download function)
  CGBN_CHECK_RETURN(stream->report);
  return NULL;
}

typedef powm_params_t<32, 4096, 5> params4096;

// create a bunch of streams and buffers suitable for running a particular kernel
inline const char* createStream(streamCreateInfo createInfo, streamData* stream) {
#ifdef TRACE
  printf("createStream (streamData)\n");
#endif
  stream->capacity = createInfo.capacity;
  stream->constantsCapacity = createInfo.constantsCapacity;
  stream->outputsCapacity = createInfo.outputsCapacity;
  stream->inputsCapacity = createInfo.inputsCapacity;
  stream->length = 0;
  stream->constantsLength = 0;
  stream->outputsLength = 0;
  stream->inputsLength = 0;
  CUDA_CHECK_RETURN(cudaStreamCreate(&(stream->stream)));
  CUDA_CHECK_RETURN(cudaMalloc(&(stream->gpuInputs), createInfo.inputsCapacity));
  CUDA_CHECK_RETURN(cudaMalloc(&(stream->gpuOutputs), createInfo.outputsCapacity));
  CUDA_CHECK_RETURN(cudaMalloc(&(stream->gpuConstants), createInfo.constantsCapacity));
  CUDA_CHECK_RETURN(cgbn_error_report_alloc(&stream->report));
  CUDA_CHECK_RETURN(cudaHostAlloc(&(stream->cpuOutputs), createInfo.outputsCapacity, cudaHostAllocDefault));

  // Both of these next buffers should only be written on the host, so they can be allocated write-combined
  CUDA_CHECK_RETURN(cudaHostAlloc(&(stream->cpuInputs), createInfo.inputsCapacity, cudaHostAllocWriteCombined));
  CUDA_CHECK_RETURN(cudaHostAlloc(&(stream->cpuConstants), createInfo.constantsCapacity, cudaHostAllocWriteCombined));

  // These events are created with timing disabled because it takes time 
  // to get the timing data, and we don't need it.
  // cudaEventBlockingSync also prevents 100% cpu usage when synchronizing on an event
  CUDA_CHECK_RETURN(cudaEventCreateWithFlags(&(stream->hostToDevice), cudaEventDisableTiming|cudaEventBlockingSync));
  CUDA_CHECK_RETURN(cudaEventCreateWithFlags(&(stream->exec), cudaEventDisableTiming|cudaEventBlockingSync));
  CUDA_CHECK_RETURN(cudaEventCreateWithFlags(&(stream->deviceToHost), cudaEventDisableTiming|cudaEventBlockingSync));

  return NULL;
}

// All the methods used in cgo should have extern "C" linkage to avoid
// implementation-specific name mangling
// This makes them more straightforward to load from the shared object
extern "C" {
  // Enqueue upload data for a powm kernel run for 4K bits
  // Stage input data by copying to the stream's constants and inputs memory before calling
  const char* upload(const uint32_t instance_count, void *stream, size_t inputsUploadSize, size_t constantsUploadSize, size_t outputsDownloadSize) {
#ifdef TRACE
    printf("upload (void)\n");
#endif
    auto gpuData = (streamData*)stream;
    // Previous download must finish before data are uploaded
    CUDA_CHECK_RETURN(cudaStreamWaitEvent(gpuData->stream, gpuData->deviceToHost, 0));
    
    // Set instance count; it's re-used when the kernel gets run later
    gpuData->length = instance_count;
    // If there are more instances uploaded than the stream can handle, that's an error
    // At least for now
    if (gpuData->length > gpuData->capacity) {
      return strdup("upload_powm error: length greater than capacity\n");
    }
    // It also doesn't take too long to bounds check the requested data transfer sizes
    // to avoid segmentation faults
    if (gpuData->inputsCapacity < inputsUploadSize) {
      return strdup("upload error: input upload size greater than capacity\n");
    }
    if (gpuData->constantsCapacity < constantsUploadSize) {
      return strdup("upload error: constants upload size greater than capacity\n");
    }
    if (gpuData->outputsCapacity < outputsDownloadSize) {
      return strdup("upload error: outputs download size greater than capacity\n");
    }

    // At this point, everything should be in a good state, so set the needed variables
    gpuData->inputsLength = inputsUploadSize;
    gpuData->outputsLength = outputsDownloadSize;
    gpuData->constantsLength = constantsUploadSize;

    CUDA_CHECK_RETURN(cudaMemcpyAsync(gpuData->gpuInputs, gpuData->cpuInputs, gpuData->inputsLength,
          cudaMemcpyHostToDevice, gpuData->stream));

    // Currently, we're copying to the constants before each kernel launch
    // This might not be necessary, but it's not that much data
    CUDA_CHECK_RETURN(cudaMemcpyAsync(gpuData->gpuConstants, gpuData->cpuConstants, gpuData->constantsLength,
          cudaMemcpyHostToDevice, gpuData->stream));

    // Run should wait on this event for kernel launch
    // The event should include any memcpys that haven't completed yet
    CUDA_CHECK_RETURN(cudaEventRecord(gpuData->hostToDevice, gpuData->stream));

    return NULL;
  }
  
  // Run powm for 4K bits
  const char* run(void *stream, kernel whichToRun) {
#ifdef TRACE
    printf("run (void)\n");
#endif
    return run<params4096>((streamData*)stream, whichToRun);
  }

  const char* download(void *s) {
#ifdef TRACE
    printf("download (void)\n");
#endif
    auto stream = (streamData*)s;
    // Wait for the kernel to finish running
    CUDA_CHECK_RETURN(cudaStreamWaitEvent(stream->stream, stream->exec, 0));

    // The kernel ran successfully, so we get the results off the GPU
    // This should cause invalid argument errors
    CUDA_CHECK_RETURN(cudaMemcpyAsync(stream->cpuOutputs, stream->gpuOutputs, stream->outputsLength, cudaMemcpyDeviceToHost, stream->stream));
    CUDA_CHECK_RETURN(cudaEventRecord(stream->deviceToHost, stream->stream));

    return NULL;
  }

  struct return_data* getResults(void *stream) {
#ifdef TRACE
    printf("getResults (void)\n");
#endif
    return_data* result = (return_data*)malloc(sizeof(*result));
    result->result = ((streamData*)stream)->cpuOutputs;
    result->error = getResults((streamData*)stream);
    return result;
  }

  // Call this when starting the program to allocate resources
  // Returns stream or error
  struct return_data* createStream(streamCreateInfo createInfo) {
#ifdef TRACE
    printf("createStream (streamCreateInfo)\n");
#endif
    return_data* result = (return_data*)malloc(sizeof(*result));
    streamData *s = (streamData*)(malloc(sizeof(*s)));
    result->error = createStream(createInfo, s);
    result->result = s;
    return result;
  }

  // Call this after execution has completed to deallocate resources
  // Returns error
  const char* destroyStream(void *destroyee) {
#ifdef TRACE
    printf("destroyStream (void)\n");
#endif
    auto stream = (streamData*)destroyee;
    // Don't know at what point there could have been errors while creating this stream,
    // so make sure things exist before destroying them
    if (stream != NULL) {
      if (stream->gpuInputs != NULL) {
        CUDA_CHECK_RETURN(cudaFree(stream->gpuInputs));
      }
      if (stream->gpuOutputs != NULL) {
        CUDA_CHECK_RETURN(cudaFree(stream->gpuOutputs));
      }
      if (stream->gpuConstants != NULL) {
        CUDA_CHECK_RETURN(cudaFree(stream->gpuConstants));
      }
      if (stream->cpuInputs != NULL) {
        CUDA_CHECK_RETURN(cudaFreeHost(stream->cpuInputs));
      }
      if (stream->cpuOutputs != NULL) {
        CUDA_CHECK_RETURN(cudaFreeHost(stream->cpuOutputs));
      }
      if (stream->cpuConstants != NULL) {
        CUDA_CHECK_RETURN(cudaFreeHost(stream->cpuConstants));
      }
      if (stream->report != NULL) {
        CUDA_CHECK_RETURN(cgbn_error_report_free(stream->report));
      }
      if (stream->hostToDevice != NULL) {
        CUDA_CHECK_RETURN(cudaEventDestroy(stream->hostToDevice));
      }
      if (stream->exec != NULL) {
        CUDA_CHECK_RETURN(cudaEventDestroy(stream->exec));
      }
      if (stream->deviceToHost != NULL) {
        CUDA_CHECK_RETURN(cudaEventDestroy(stream->deviceToHost));
      }
      free((void*)stream);
    }

    return NULL;
  }

  // Call this after execution has completed to write out profile information to the disk
  const char* stopProfiling() {
    CUDA_CHECK_RETURN(cudaProfilerStop());
    return NULL;
  }

  const char* startProfiling() {
    CUDA_CHECK_RETURN(cudaProfilerStart());
    return NULL;
  }

  const char* resetDevice() {
    CUDA_CHECK_RETURN(cudaDeviceReset());
    return NULL;
  }

  // Return cpu inputs buffer pointer for writing
  void* getCpuInputs(void* stream) {
    return ((streamData*)stream)->cpuInputs;
  }

  // Return cpu outputs buffer pointer for reading
  void* getCpuOutputs(void* stream) {
    return ((streamData*)stream)->cpuOutputs;
  }

  // Return cpu constants buffer pointer for writing
  void* getCpuConstants(void* stream) {
    return ((streamData*)stream)->cpuConstants;
  }
}

