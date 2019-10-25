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

struct streamManager {
  uint32_t numStreams;
  uint32_t currentStreamIndex;
  streamData *streams;
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
class powm_odd_t {
  public:
  static const uint32_t window_bits=params::WINDOW_BITS;  // used a lot, give it an instance variable

  // It might be possible to switch to a SOA structure within the instance_t struct
  // Currently, I believe removing this struct completely would make things worse
  // The main advantage of the current interleaved AOS input structure is that it allows making the 
  // input memory longer by concatenating byte arrays that represent valid inputs
  // I also need to run benchmarks on an x16 pcie link to make sure we're making the correct pcie bandwidth tradeoff
  // Results shouldn't belong in the instance struct. They should get allocated and written separately, so as to not
  // have to download and uploaded more than is necessary. x and pow should only be uploaded, and results should only
  // be downloaded.
  typedef struct {
    cgbn_mem_t<params::BITS> x;
    cgbn_mem_t<params::BITS> power;
  } input_t;
  
  typedef cgbn_context_t<params::TPI, params>   context_t;
  typedef cgbn_env_t<context_t, params::BITS>   env_t;
  typedef typename env_t::cgbn_t                bn_t;
  typedef typename env_t::cgbn_local_t          bn_local_t;

  context_t _context;
  env_t     _env;
  int32_t   _instance;

  __device__ __forceinline__ powm_odd_t(cgbn_monitor_t monitor, cgbn_error_report_t *report, int32_t instance) : _context(monitor, report, (uint32_t)instance), _env(_context), _instance(instance) {
  }

  __device__ __forceinline__ void fixed_window_powm_odd(bn_t &result, const bn_t &x, const bn_t &power, const bn_t &modulus) {
    bn_t       t;
    bn_local_t window[1<<window_bits];
    int32_t    index, position, offset;
    uint32_t   np0;

    // conmpute x^power mod modulus, using the fixed window algorithm
    // requires:  x<modulus,  modulus is odd

    // compute x^0 (in Montgomery space, this is just 2^BITS - modulus)
    cgbn_negate(_env, t, modulus);
    cgbn_store(_env, window+0, t);
    
    // convert x into Montgomery space, store into window table
    np0=cgbn_bn2mont(_env, result, x, modulus);
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
    
    // we've processed the exponent now, convert back to normal space
    cgbn_mont2bn(_env, result, result, modulus, np0);
  }
  
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

  // Convenience methods to calculate required size for inputs, outputs, and constants
  __host__ static inline size_t getInputsSize(size_t length) {
    // Each input is the size of an instance
    return sizeof(input_t) * length;
  }

  __host__ static inline size_t getOutputsSize(size_t length) {
    // There's only one number in each output
    return sizeof(cgbn_mem_t<params::BITS>) * length;
  }

  __host__ static inline size_t getConstantsSize() {
    // There's one constant, the modulus
    return sizeof(cgbn_mem_t<params::BITS>);
  }
};

// kernel implementation using cgbn
// 
// Unfortunately, the kernel must be separate from the powm_odd_t class
// kernel_powm_odd<params><<<(instance_count+IPB-1)/IPB, TPB>>>(report, gpuInputs, gpuResults, instance_count);
template<class params>
__global__ void kernel_powm_odd(cgbn_error_report_t *report, typename powm_odd_t<params>::input_t *inputs, cgbn_mem_t<params::BITS> *modulus, cgbn_mem_t<params::BITS> *outputs, size_t count) {
  int32_t instance;

  // decode an instance number from the blockIdx and threadIdx
  instance=(blockIdx.x*blockDim.x + threadIdx.x)/params::TPI;
  if(instance>=count)
    return;

  powm_odd_t<params>                 po(cgbn_report_monitor, report, instance);
  typename powm_odd_t<params>::bn_t  r, x, p, m;
  
  // the loads and stores can go in the class, but it seems more natural to have them
  // here and to pass in and out bignums
  cgbn_load(po._env, x, &(inputs[instance].x));
  cgbn_load(po._env, p, &(inputs[instance].power));
  cgbn_load(po._env, m, modulus);
  
  // this can be either fixed_window_powm_odd or sliding_window_powm_odd.
  // when TPI<32, fixed window runs much faster because it is less divergent, so we use it here
  po.fixed_window_powm_odd(r, x, p, m);
  //   OR
  // po.sliding_window_powm_odd(r, x, p, m);
  
  cgbn_store(po._env, &(outputs[instance]), r);
}

// Enqueue a host-to-device transfer before a kernel run
template<class params>
const char* upload_powm(const uint32_t instance_count, streamData* gpuData) {
  typedef typename powm_odd_t<params>::input_t input_t;

  // Previous download must finish before data are uploaded
  CUDA_CHECK_RETURN(cudaStreamWaitEvent(gpuData->stream, gpuData->deviceToHost, 0));
  
  // Set instance count; it's re-used when the kernel gets run later
  gpuData->length = instance_count;
  // If there are more instances uploaded than the stream can handle, that's an error
  // At least for now
  if (gpuData->length > gpuData->capacity) {
    return strdup("upload_powm error: length greater than capacity\n");
  }

  CUDA_CHECK_RETURN(cudaMemcpyAsync(gpuData->gpuInputs, gpuData->cpuInputs,
        powm_odd_t<params>::getInputsSize(gpuData->length),
        cudaMemcpyHostToDevice, gpuData->stream));

  // Currently, we're copying to the modulus before each kernel launch
  // This should technically be unnecessary, but it's not that much data
  CUDA_CHECK_RETURN(cudaMemcpyAsync(gpuData->gpuConstants, gpuData->cpuConstants, 
        powm_odd_t<params>::getConstantsSize(), cudaMemcpyHostToDevice, gpuData->stream));

  // Run should wait on this event for kernel launch
  // The event should include any memcpys that haven't completed yet
  CUDA_CHECK_RETURN(cudaEventRecord(gpuData->hostToDevice, gpuData->stream));

  return NULL;
}

// Run powm kernel
// Enqueues kernel on the stream and returns immediately (non-blocking)
// To call this, you should have prepared a kernel launch with upload_powm
// and waited for the returned struct to be populated
// The method will only work properly with a valid (i.e. non-error) 
// powm_upload_results_t
// The results will be placed in the passed results pointer after the kernel run
// Precondition: stream should have had upload called.
template<class params>
const char* run_powm(streamData *stream) {
  // TODO Wait on upload event to finish before running kernel
  //  Can't be done until we switch to async uploads
  typedef typename powm_odd_t<params>::input_t input_t;
  typedef cgbn_mem_t<params::BITS> num_t;

  const int32_t              TPB=(params::TPB==0) ? 128 : params::TPB;    // default threads per block to 128
  const int32_t              TPI=params::TPI, IPB=TPB/TPI;                // IPB is instances per block

  CUDA_CHECK_RETURN(cudaStreamWaitEvent(stream->stream, stream->hostToDevice, 0));
  // launch kernel with blocks=ceil(instance_count/IPB) and threads=TPB
  kernel_powm_odd<params><<<(stream->length+IPB-1)/IPB, TPB, 0, stream->stream>>>(
    stream->report, 
    (input_t*)stream->gpuInputs, 
    (num_t*)stream->gpuConstants, 
    (num_t*)stream->gpuOutputs, 
    stream->length);

  // error report uses managed memory, so we sync the device (or stream) and check for cgbn errors
  // Note: This should probably only happen in debug builds, as the error 
  // report might not be necessary in normal usage
  CUDA_CHECK_RETURN(cudaEventRecord(stream->exec, stream->stream));

  return NULL;
}

// Currently, the download blocks
// It's possible to make this block separately
template<class params>
// Enqueue a download of the results after the kernel finishes running
const char* download_powm(streamData *stream) {
  // Wait for the kernel to finish running
  CUDA_CHECK_RETURN(cudaStreamWaitEvent(stream->stream, stream->exec, 0));

  // The kernel ran successfully, so we get the results off the GPU
  CUDA_CHECK_RETURN(cudaMemcpyAsync(stream->cpuOutputs, stream->gpuOutputs, powm_odd_t<params>::getOutputsSize(stream->length), cudaMemcpyDeviceToHost, stream->stream));
  CUDA_CHECK_RETURN(cudaEventRecord(stream->deviceToHost, stream->stream));

  return NULL;
}

const char* getResults_powm(streamData *stream) {
  // Wait for download to complete
  CUDA_CHECK_RETURN(cudaEventSynchronize(stream->deviceToHost));
  CGBN_CHECK_RETURN(stream->report);
  return NULL;
}

typedef powm_params_t<32, 4096, 5> params4096;

template<class params>
inline const char* run_powm_export(streamData *stream) {
  return run_powm<params>(stream);
}

template<class params>
inline const char* upload_export(const uint32_t instance_count, streamData *stream) {
  return upload_powm<params>(instance_count, stream);
}

template<class params>
inline const char* download_powm_export(streamData *stream) {
  return download_powm<params>(stream);
}

inline return_data* getResults_powm_export(streamData *stream) {
  return_data* result = (return_data*)malloc(sizeof(*result));
  result->result = stream->cpuOutputs;
  result->error = getResults_powm(stream);
  return result;
}


// create a bunch of streams and buffers suitable for running a particular kernel
const char* createStreamManager(streamManagerCreateInfo createInfo, streamManager *streams) {
  streams->numStreams = createInfo.numStreams;
  streams->streams = (typeof(streams->streams))malloc(sizeof(*streams->streams)*createInfo.numStreams);

  for (int i = 0; i < createInfo.numStreams; i++) {
    CUDA_CHECK_RETURN(cudaStreamCreate(&streams->streams[i].stream));
    CUDA_CHECK_RETURN(cudaMalloc(&(streams->streams[i].gpuInputs), createInfo.inputsSize));
    CUDA_CHECK_RETURN(cudaMalloc(&(streams->streams[i].gpuOutputs), createInfo.outputsSize));
    CUDA_CHECK_RETURN(cudaMalloc(&(streams->streams[i].gpuConstants), createInfo.constantsSize));
    streams->streams[i].capacity = createInfo.capacity;
    streams->streams[i].length = 0;
    CUDA_CHECK_RETURN(cgbn_error_report_alloc(&streams->streams[i].report));
    CUDA_CHECK_RETURN(cudaHostAlloc(&(streams->streams[i].cpuOutputs), createInfo.outputsSize, cudaHostAllocDefault));

    // Both of these next buffers should only be written on the host, so they can be allocated write-combined
    CUDA_CHECK_RETURN(cudaHostAlloc(&(streams->streams[i].cpuInputs), createInfo.inputsSize, cudaHostAllocWriteCombined));
    CUDA_CHECK_RETURN(cudaHostAlloc(&(streams->streams[i].cpuConstants), createInfo.constantsSize, cudaHostAllocWriteCombined));

    // These events are created with timing disabled because it takes time 
    // to get the timing data, and we don't need it.
    CUDA_CHECK_RETURN(cudaEventCreateWithFlags(&(streams->streams[i].hostToDevice), cudaEventDisableTiming|cudaEventBlockingSync));
    CUDA_CHECK_RETURN(cudaEventCreateWithFlags(&(streams->streams[i].exec), cudaEventDisableTiming|cudaEventBlockingSync));
    CUDA_CHECK_RETURN(cudaEventCreateWithFlags(&(streams->streams[i].deviceToHost), cudaEventDisableTiming|cudaEventBlockingSync));
  }
  return NULL;
}

// allocate the data necessary to return a stream manager with an error across the bindings
return_data* createStreamManager_export(streamManagerCreateInfo createInfo) {
  return_data* result = (return_data*)malloc(sizeof(*result));
  streamManager *sm = (streamManager*)(malloc(sizeof(*sm)));
  result->error = createStreamManager(createInfo, sm);
  result->result = sm;
  return result;
}

// free the space used by a stream manager that's no longer in use
inline const char* destroyStreamManager(streamManager *streams) {
  // After the program's done with this kernel, the stream manager is no 
  // longer needed, and should get cleaned up.
  for (int i = 0; i < streams->numStreams; i++) {
    CUDA_CHECK_RETURN(cudaFree(streams->streams[i].gpuInputs));
    CUDA_CHECK_RETURN(cudaFree(streams->streams[i].gpuOutputs));
    CUDA_CHECK_RETURN(cudaFree(streams->streams[i].gpuConstants));
    CUDA_CHECK_RETURN(cudaFreeHost(streams->streams[i].cpuInputs));
    CUDA_CHECK_RETURN(cudaFreeHost(streams->streams[i].cpuOutputs));
    CUDA_CHECK_RETURN(cudaFreeHost(streams->streams[i].cpuConstants));
    CUDA_CHECK_RETURN(cgbn_error_report_free(streams->streams[i].report));
    CUDA_CHECK_RETURN(cudaEventDestroy(streams->streams[i].hostToDevice));
    CUDA_CHECK_RETURN(cudaEventDestroy(streams->streams[i].exec));
    CUDA_CHECK_RETURN(cudaEventDestroy(streams->streams[i].deviceToHost));
  }
  free((void*)streams);
  return NULL;
}

// All the methods used in cgo should have extern "C" linkage to avoid
// implementation-specific name mangling
// This makes them more straightforward to load from the shared object
extern "C" {
  // Enqueue upload data for a powm kernel run for 4K bits
  // Stage input data by copying to the stream's constants and inputs memory before calling
  const char* upload_powm_4096(const uint32_t instance_count, void *stream) {
    return upload_export<params4096>(instance_count, (streamData*)stream);
  }
  
  // Run powm for 4K bits
  const char* run_powm_4096(void *stream) {
    return run_powm_export<params4096>((streamData*)stream);
  }

  const char* download_powm_4096(void *stream) {
    return download_powm_export<params4096>((streamData*)stream);
  }

  struct return_data* getResults_powm(void *stream) {
    return getResults_powm_export((streamData*)stream);
  }

  // Call this when starting the program to allocate resources
  // Returns pointer to class and error
  struct return_data* createStreamManager(streamManagerCreateInfo createInfo) {
    return createStreamManager_export(createInfo);
  }

  // Call this after execution has completed to deallocate resources
  // Returns error
  const char* destroyStreamManager(void *destroyee) {
    return destroyStreamManager((streamManager*)(destroyee));
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

  size_t getConstantsSize_powm4096() {
    return powm_odd_t<params4096>::getConstantsSize();
  }

  size_t getInputsSize_powm4096(size_t length) {
    return powm_odd_t<params4096>::getInputsSize(length);
  }
  
  size_t getOutputsSize_powm4096(size_t length) {
    return powm_odd_t<params4096>::getOutputsSize(length);
  }

  // These are void pointers instead of real types, because the Golang bindings shouldn't
  // be doing anything with them other than passing them to this library
  void* getNextStream(void* pStreamManager) {
    streamManager *sm = (streamManager*)pStreamManager;
    uint32_t nextStreamIndex = (sm->currentStreamIndex + 1) % sm->numStreams;
    sm->currentStreamIndex = nextStreamIndex;
    return (void*)(&sm->streams[nextStreamIndex]);
  }

  // Return cpu inputs buffer pointer for writing
  void* getCpuInputs(void* stream) {
    streamData *gpuData = (streamData*)(stream);
    return gpuData->cpuInputs;
  }

  // Return cpu outputs buffer pointer for reading
  void* getCpuOutputs(void* stream) {
    streamData *gpuData = (streamData*)(stream);
    return gpuData->cpuOutputs;
  }

  // Return cpu constants buffer pointer for writing
  void* getCpuConstants(void* stream) {
    streamData *gpuData = (streamData*)(stream);
    return gpuData->cpuConstants;
  }
}

