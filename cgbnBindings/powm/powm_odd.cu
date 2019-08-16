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

// Is there no way to use the class parameters to populate the length of this array?
// It's a CUDA limitation that __constant__ variables can't be members of a class...
// We could go with the old-school macro route.
static __constant__ cgbn_mem_t<2048> MODULUS;    // modulus is the same for all instances, at least during each kernel launch
                                                 // TODO(nan) Is it a good idea to re-upload the modulus each time?
                                                 // It's not that much bandwidth required...
                                                 // how about running the kernel on a smaller and a bigger prime?
                                                 // what would be the best way to set that up?
                                                 // Do you just need to have two global variables, one for each modulus?
                                                 // Maybe the additional limitations that constant variables give aren't worth the extra headache.

template<class params>
class powm_odd_t {
  public:
  static const uint32_t window_bits=params::WINDOW_BITS;  // used a lot, give it an instance variable

  // It might be possible to switch to a SOA structure within the instance_t struct
  // Currently, I believe removing this struct completely would make things worse
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
    // Conclusion: Right now, only one byte is getting copied by cudaMemcpyToSymbol.
    //  This will of course result in an error, as there aren't enough bits
    //  available to do a Montgomery reduction.
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
__global__ void kernel_powm_odd(cgbn_error_report_t *report, typename powm_odd_t<params>::input_t *inputs, cgbn_mem_t<params::BITS> *outputs, uint32_t count) {
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
  cgbn_load(po._env, m, &MODULUS);
  
  // this can be either fixed_window_powm_odd or sliding_window_powm_odd.
  // when TPI<32, fixed window runs much faster because it is less divergent, so we use it here
  po.fixed_window_powm_odd(r, x, p, m);
  //   OR
  // po.sliding_window_powm_odd(r, x, p, m);
  
  cgbn_store(po._env, &(outputs[instance]), r);
}

// TODO Is there a way to be type safe without an array of structs setup?
// Returns error
template<class params>
const char* run_powm(const void* modulus, const void *inputs, void *results, const uint32_t instance_count) {
  typedef typename powm_odd_t<params>::input_t input_t;
  // TODO Each kernel run should return all errors that occurred during the run in a single string
  
  cgbn_error_report_t *report;
  int32_t              TPB=(params::TPB==0) ? 128 : params::TPB;    // default threads per block to 128
  int32_t              TPI=params::TPI, IPB=TPB/TPI;                // IPB is instances per block
  input_t *gpuInputs;
  cgbn_mem_t<params::BITS> *gpuResults;
  const char *err = NULL;
  
  err = CUDA_CHECK(cudaSetDevice(0));
  RETURN_IF_EXISTS(err);
  printf("Copying inputs to the GPU ...\n");
  // Is this the best way of allocating memory for each kernel launch?
  // Is there actually a perf difference doing things this way vs the AoS allocation style?
  // Results will be written to the end of this area of memory
  // I'm pretty sure this is a dumb way of doing it...
  size_t modulusSize = sizeof(cgbn_mem_t<params::BITS>);
  const size_t resultsSize = sizeof(cgbn_mem_t<params::BITS>)*instance_count;
  const size_t inputsSize = sizeof(input_t)*instance_count;
  err = CUDA_CHECK(cudaMalloc((void **)&gpuInputs, inputsSize));
  RETURN_IF_EXISTS(err);
  err = CUDA_CHECK(cudaMalloc((void **)&gpuResults, resultsSize));
  RETURN_IF_EXISTS(err);

  err = CUDA_CHECK(cudaMemcpy(gpuInputs, inputs, inputsSize, cudaMemcpyHostToDevice));
  RETURN_IF_EXISTS(err);

  // Currently, we're copying to the modulus before each kernel launch
  // I'm not sure how to handle benchmarking with two groups...
  printf("Copying modulus to the GPU ...\n");
  err = CUDA_CHECK(cudaMemcpyToSymbol(MODULUS, modulus, modulusSize, 0, cudaMemcpyHostToDevice));
  RETURN_IF_EXISTS(err);

  // create a cgbn_error_report for CGBN to report back errors
  err = CUDA_CHECK(cgbn_error_report_alloc(&report));
  RETURN_IF_EXISTS(err);

  printf("Running GPU kernel ...\n");
  
  // launch kernel with blocks=ceil(instance_count/IPB) and threads=TPB
  kernel_powm_odd<params><<<(instance_count+IPB-1)/IPB, TPB>>>(report, gpuInputs, gpuResults, instance_count);

  // error report uses managed memory, so we sync the device (or stream) and check for cgbn errors
  err = CUDA_CHECK(cudaDeviceSynchronize());
  RETURN_IF_EXISTS(err);
  err = CGBN_CHECK(report);
  RETURN_IF_EXISTS(err);

  // copy the results back from gpuMemory
  printf("Copying results back to CPU ...\n");
  // We don't actually need to memcpy anything that's not an output
  err = CUDA_CHECK(cudaMemcpy(results, gpuResults, resultsSize, cudaMemcpyDeviceToHost));
  RETURN_IF_EXISTS(err);

  // clean up
  // TODO Instances will now need to be freed manually from the Go side once
  //  GC-tracked copies are made. We will need a new method that calls free on this memory.
  err = CUDA_CHECK(cudaFree(gpuInputs));
  RETURN_IF_EXISTS(err);
  err = CUDA_CHECK(cudaFree(gpuResults));
  RETURN_IF_EXISTS(err);
  err = CUDA_CHECK(cgbn_error_report_free(report));
  RETURN_IF_EXISTS(err);
  return NULL;
}

// All the methods used in cgo should have extern "C" linkage to avoid
// implementation-specific name mangling
// This makes them more straightforward to load from the shared object
extern "C" {
    // Can the 2048 be templated?
    typedef powm_params_t<8, 2048, 5> params;
    struct powm_2048_return {
        void *powm_results;
        const char *error;
    };
    powm_2048_return* powm_2048(const void *prime, const void *instances, const uint32_t instance_count) {
        printf("Error is after CUDA so call\n");
        powm_2048_return *result = (powm_2048_return*)malloc(sizeof(struct powm_2048_return));
        // Can i get the size of an individual BN in a better way than this?
        void *results_mem = malloc(sizeof(params::BITS/8 * instance_count));
        result->error = run_powm<params>(prime, instances, results_mem, instance_count);
        result->powm_results = results_mem;
        return result;
    }
}
