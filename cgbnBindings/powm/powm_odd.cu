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
};

// kernel implementation using cgbn
// 
// Unfortunately, the kernel must be separate from the powm_odd_t class
// kernel_powm_odd<params><<<(instance_count+IPB-1)/IPB, TPB>>>(report, gpuInputs, gpuResults, instance_count);
template<class params>
__global__ void kernel_powm_odd(cgbn_error_report_t *report, typename powm_odd_t<params>::input_t *inputs, cgbn_mem_t<params::BITS> *modulus, cgbn_mem_t<params::BITS> *outputs, uint32_t count) {
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

// Result of upload_powm
template<class params>
struct powm_upload_results_t {
  // Number of items: instance_count
  typename powm_odd_t<params>::input_t *gpuInputs;
  cgbn_mem_t<params::BITS> *gpuResults;
  // Number of items: 1
  cgbn_mem_t<params::BITS> *gpuModulus;
  uint32_t instance_count;
  cgbn_error_report_t *report;
  // If error is set, caller should cudaFree any GPU memory, then free the 
  // error string and the struct when the error has been handled.
  const char* error;
};

// Check error before proceeding
// Does async memcpy set the error?
// Clean up struct if error is present
// Uploads memory from host to device, asynchronously
// Returns a struct that will contain the necessary parameters to the run function
template<class params>
powm_upload_results_t<params>* upload_powm(const void* modulus, const void *inputs, const uint32_t instance_count) {
  typedef typename powm_odd_t<params>::input_t input_t;
  
  powm_upload_results_t<params>* result = (powm_upload_results_t<params>*)malloc(sizeof(*result));
  // Set instance count; it's re-used when the kernel gets run later
  result->instance_count = instance_count;
  
  // Because there aren't multiple return types, this will no longer work
  result->error = CUDA_CHECK(cudaSetDevice(0));
  if (result->error != NULL) {
    return result;
  }
  printf("Copying inputs to the GPU ...\n");
  // Is this the best way of allocating memory for each kernel launch?
  // Is there actually a perf difference doing things this way vs the AoS allocation style?
  // Results will be written to the end of this area of memory
  // I'm pretty sure this is a dumb way of doing it...
  const size_t modulusSize = sizeof(cgbn_mem_t<params::BITS>);
  const size_t resultsSize = sizeof(cgbn_mem_t<params::BITS>)*instance_count;
  const size_t inputsSize = sizeof(input_t)*instance_count;
  result->error = CUDA_CHECK(cudaMalloc((void **)&(result->gpuInputs), inputsSize));
  if (result->error != NULL) {
    return result;
  }
  result->error = CUDA_CHECK(cudaMalloc((void **)&(result->gpuResults), resultsSize));
  if (result->error != NULL) {
    return result;
  }
  result->error = CUDA_CHECK(cudaMalloc((void **)&(result->gpuModulus), modulusSize));
  if (result->error != NULL) {
    return result;
  }

  result->error = CUDA_CHECK(cudaMemcpy((void *)result->gpuInputs, inputs, inputsSize, cudaMemcpyHostToDevice));
  if (result->error != NULL) {
    return result;
  }

  // Currently, we're copying to the modulus before each kernel launch
  // I'm not sure how to handle benchmarking with two groups...
  // Two groups would require two separate kernel launches, and if they have
  // different numbers of bits, they would also require two different CGBN
  // environments... Basically, two different instantiations of the class.
  result->error = CUDA_CHECK(cudaMemcpy((void *)result->gpuModulus, modulus, modulusSize, cudaMemcpyHostToDevice));
  if (result->error != NULL) {
    return result;
  }

  // create a cgbn_error_report for CGBN to report back errors
  result->error = CUDA_CHECK(cgbn_error_report_alloc(&(result->report)));
  if (result->error != NULL) {
    return result;
  }

  return result;
}

// Run powm kernel
// Blocks until kernel execution finishes, then copies results from device to host
// To call this, you should have prepared a kernel launch with upload_powm
// and waited for the returned struct to be populated
// The method will only work properly with a valid (i.e. non-error) 
// powm_upload_results_t
// The results will be placed in the passed results pointer after the kernel run
template<class params>
const char* run_powm(powm_upload_results_t<params> *upload, void *results) {
  typedef typename powm_odd_t<params>::input_t input_t;

  const int32_t              TPB=(params::TPB==0) ? 128 : params::TPB;    // default threads per block to 128
  const int32_t              TPI=params::TPI, IPB=TPB/TPI;                // IPB is instances per block

  const size_t resultsSize = sizeof(cgbn_mem_t<params::BITS>)*upload->instance_count;

  // launch kernel with blocks=ceil(instance_count/IPB) and threads=TPB
  // We'll try a launch with just 1 instance, and see if that access is still illegal
  // Probably the memory is not getting uploaded all in one chunk, as it should be.
  kernel_powm_odd<params><<<(upload->instance_count+IPB-1)/IPB, TPB>>>(upload->report, upload->gpuInputs, upload->gpuModulus, upload->gpuResults, upload->instance_count);

  // error report uses managed memory, so we sync the device (or stream) and check for cgbn errors
  // Note: This should probably only happen in debug builds, as the error 
  // report might not be necessary in normal usage
  CUDA_CHECK_RETURN(cudaDeviceSynchronize());
  CGBN_CHECK_RETURN(upload->report);

  // The kernel ran successfully, so we get the results off the GPU
  CUDA_CHECK_RETURN(cudaMemcpy(results, upload->gpuResults, resultsSize, cudaMemcpyDeviceToHost));

  // We don't need these GPU buffers anymore, as the kernel has run
  // Does this free the buffers properly? I have concerns about correctness here
  CUDA_CHECK_RETURN(cudaFree((void*)upload->gpuInputs));
  CUDA_CHECK_RETURN(cudaFree((void*)upload->gpuResults));
  CUDA_CHECK_RETURN(cudaFree((void*)upload->gpuModulus));
  CUDA_CHECK_RETURN(cgbn_error_report_free(upload->report));
  free(upload);
  return NULL;
}

// All the methods used in cgo should have extern "C" linkage to avoid
// implementation-specific name mangling
// This makes them more straightforward to load from the shared object
extern "C" {
    // 2K BITS
    // Can the 2048 be templated?
/*    typedef powm_params_t<8, 2048, 5> params;
    struct powm_2048_return {
        void *powm_results;
        const char *error;
    };
    powm_2048_return* powm_2048(const void *prime, const void *instances, const uint32_t instance_count) {
        printf("Error is after CUDA so call\n");
        powm_2048_return *result = (powm_2048_return*)malloc(sizeof(*result));
        // Can i get the size of an individual BN in a better way than this?
        void *results_mem = malloc(params::BITS/8 * instance_count);
        result->error = run_powm<params>(prime, instances, results_mem, instance_count);
        result->powm_results = results_mem;
        return result;
    }*/

    // 4K BITS
    typedef powm_params_t<32, 4096, 5> params;
    struct powm_4096_return {
        void *powm_results;
        const char *error;
    };
    powm_4096_return* powm_4096(const void *prime, const void *instances, const uint32_t instance_count) {
        // Upload data
        auto upload = upload_powm<params>(prime, instances, instance_count);

        // Run kernel
        powm_4096_return *result = (powm_4096_return*)malloc(sizeof(*result));
        // Can i get the size of an individual BN in a better way than this?
        void *results_mem = malloc(params::BITS/8 * instance_count);
        result->error = run_powm<params>(upload, results_mem);
        result->powm_results = results_mem;
        return result;
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
}

