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

// Driver API definitions
// TODO structure this all some more perhaps
CUcontext cuContext;
CUmodule cuModule;
CUfunction powm_4096_function;
CUfunction powm_3200_function;
CUfunction powm_2048_function;
CUfunction elgamal_4096_function;
CUfunction elgamal_3200_function;
CUfunction elgamal_2048_function;
CUfunction reveal_4096_function;
CUfunction reveal_3200_function;
CUfunction reveal_2048_function;
CUfunction mul2_4096_function;
CUfunction mul2_3200_function;
CUfunction mul2_2048_function;
CUfunction mul3_4096_function;
CUfunction mul3_3200_function;
CUfunction mul3_2048_function;

// Stream object and associated data for a stream
// This name could perhaps be better...
struct streamData {
  // This CUDA stream; when performing operations, set the current stream to
  // this one
  CUstream stream;

  // Area of device memory that this stream can use
  CUdeviceptr gpuMem;
  
  // Area of host memory that this stream can use
  // This buffer is pinned memory
  void* cpuMem;

  // Total size of buffers at max capacity (set at creation time)
  size_t memCapacity;

  // Size of input and output regions for this launch (set at upload time)
  // Input region will only have host to device transfer, output region will only have device to host transfer
  // Input region always comes first, and output region always comes right afterwards
  size_t inputsLength;
  size_t outputsLength;

  enum kernel whichToRun;

  // Number of items to be processed with this part of the stream
  size_t length;

  // Check for CGBN errors after kernel finishes using this
  cgbn_error_report_t *report;
  // This event is used, along with deviceToHost, to determine the total run
  // time of this launch, including upload, download, and execution
  CUevent start;
  // Synchronize this event to wait for host to device transfer before kernel execution
  CUevent hostToDevice;
  // Synchronize this event to wait for kernel execution to finish before device to host transfer
  CUevent exec;
  // Synchronize this event to wait for downloading to finish before using results
  CUevent deviceToHost;
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

// Really I'd like this class to have very few responsibilities overall, and be able to invoke an exponentiation in the same way I'd invoke any of the native cgbn methods
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
    mem_t precomputation;
    mem_t cypher;
  } strip_input_t;
  
  typedef struct {
    mem_t x;
    mem_t y;
  } mul2_input_t;

  typedef struct {
    mem_t x;
    mem_t y;
    mem_t z;
  } mul3_input_t;

  typedef struct {
    mem_t privateKey; // Used to calculate both outputs 
    mem_t key; // Used to calculate ecrKeys output 
    mem_t ecrKeys; // Used to calculate ecrKeys output
    mem_t cypher; // Used to calculate cypher output
  } elgamal_input_t;

  typedef struct {
    mem_t ecrKeys;
    mem_t cypher;
  } elgamal_output_t;

  typedef struct {
    mem_t g;
    mem_t prime;
    mem_t publicCypherKey;
  } elgamal_constant_t;

  typedef struct {
    mem_t prime;
    mem_t Z;
  } reveal_constant_t;

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

  // Find a modular root
  // Precondition: Z is coprime to prime - 1
  __device__ __forceinline__ bool root_coprime(bn_t &result, const bn_t &cypher, const bn_t &Z, const bn_t &prime) {
    bn_t psub1, cypherMont;
    // prime should always be large, so don't check return value
    cgbn_sub_ui32(_env, psub1, prime, uint32_t(1));
    bool ok = cgbn_modular_inverse(_env, result, Z, psub1);
    if (ok) {
      // Found inverse successfully, so do the exponentiation
      uint32_t np0 = cgbn_bn2mont(_env, cypherMont, cypher, prime);
      fixed_window_powm_odd(cypherMont, cypherMont, result, prime, np0);
      cgbn_mont2bn(_env, result, cypherMont, prime, np0);
    } else {
      // The inversion result was undefined, so we must report an error
      _context.report_error(cgbn_inverse_does_not_exist_error);
    }
    return ok;
  }
};

// kernel implementation using cgbn
// 
// Unfortunately, the kernel must be separate from the cmixPrecomp class
// kernel_powm_odd<params><<<(instance_count+IPB-1)/IPB, TPB>>>(report, gpuInputs, gpuResults, instance_count);
template<class params>
__device__ void kernel_powm_odd(cgbn_error_report_t *report, typename cmixPrecomp<params>::mem_t *constants, typename cmixPrecomp<params>::powm_odd_input_t *inputs, typename cmixPrecomp<params>::mem_t *outputs, size_t count) {
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
  cgbn_load(po._env, m, constants);
  
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
__device__ void kernel_elgamal(cgbn_error_report_t *report, typename cmixPrecomp<params>::elgamal_constant_t *constants, typename cmixPrecomp<params>::elgamal_input_t *inputs, typename cmixPrecomp<params>::elgamal_output_t *outputs, size_t count) {
  int32_t instance;

  // decode an instance number from the blockIdx and threadIdx
  instance=(blockIdx.x*blockDim.x + threadIdx.x)/params::TPI;
  if(instance>=count)
    return;

  cmixPrecomp<params>                 po(cgbn_report_monitor, report, instance);
  typename cmixPrecomp<params>::bn_t privateKey, ecrKeys, key, publicCypherKey, cypher, g, prime, result;

  // Prepare elgamal inputs
  cgbn_load(po._env, privateKey, &(inputs[instance].privateKey));
  cgbn_load(po._env, key, &(inputs[instance].key));
  cgbn_load(po._env, publicCypherKey, &(constants->publicCypherKey));
  cgbn_load(po._env, ecrKeys, &(inputs[instance].ecrKeys));
  cgbn_load(po._env, cypher, &(inputs[instance].cypher));
  cgbn_load(po._env, g, &(constants->g));
  cgbn_load(po._env, prime, &(constants->prime));
  // TODO load/calculate np0 if not montgomery converting
  // It's also possible to always convert g and publicCypherKey to mont space before uploading!
  // This would save a little time to not convert them all the time. But it requires making mont
  // conversion routines on CPU. Not sure if golang big int library can do that. If not it would
  // be straightforward to write. But you'd get g, publicCypherKey, and np0 ready 
  // on all instances for cheap.

  // Calculate ecrKeys first
  // TODO experiment with the ordering of the calls here. It may be possible to gain some speed by rearranging things
  uint32_t np0 = cgbn_bn2mont(po._env, g, g, prime);
  po.fixed_window_powm_odd(result, g, privateKey, prime, np0);
  cgbn_bn2mont(po._env, key, key, prime);
  cgbn_mont_mul(po._env, result, result, key, prime, np0);
  cgbn_bn2mont(po._env, ecrKeys, ecrKeys, prime);
  cgbn_mont_mul(po._env, result, result, ecrKeys, prime, np0);
  cgbn_mont2bn(po._env, result, result, prime, np0);
  cgbn_store(po._env, &(outputs[instance].ecrKeys), result);

  // Calculate cypher second
  cgbn_bn2mont(po._env, publicCypherKey, publicCypherKey, prime);
  po.fixed_window_powm_odd(result, publicCypherKey, privateKey, prime, np0);
  cgbn_bn2mont(po._env, cypher, cypher, prime);
  cgbn_mont_mul(po._env, result, result, cypher, prime, np0);
  cgbn_mont2bn(po._env, result, result, prime, np0);
  cgbn_store(po._env, &(outputs[instance].cypher), result);
}

template<class params>
__device__ void kernel_reveal(cgbn_error_report_t *report, typename cmixPrecomp<params>::reveal_constant_t *constants, typename cmixPrecomp<params>::mem_t *inputs, typename cmixPrecomp<params>::mem_t *outputs, size_t count) {
  int32_t instance;

  // decode an instance number from the blockIdx and threadIdx
  instance=(blockIdx.x*blockDim.x + threadIdx.x)/params::TPI;
  if(instance>=count)
    return;

  cmixPrecomp<params>                 po(cgbn_report_monitor, report, instance);
  typename cmixPrecomp<params>::bn_t  cypher, Z, prime, result;

  cgbn_load(po._env, cypher, &(inputs[instance]));
  cgbn_load(po._env, Z, &(constants->Z));
  cgbn_load(po._env, prime, &(constants->prime));

  po.root_coprime(result, cypher, Z, prime);

  cgbn_store(po._env, &(outputs[instance]), result);
}

// Multiply x by y mod prime
template<class params>
__device__ void kernel_mul2(cgbn_error_report_t *report, typename cmixPrecomp<params>::mem_t *constants, typename cmixPrecomp<params>::mul2_input_t *inputs, typename cmixPrecomp<params>::mem_t *outputs, size_t count) {
  int32_t instance;

  // decode an instance number from the blockIdx and threadIdx
  instance=(blockIdx.x*blockDim.x + threadIdx.x)/params::TPI;
  if(instance>=count)
    return;

  cmixPrecomp<params>                 po(cgbn_report_monitor, report, instance);
  typename cmixPrecomp<params>::bn_t  x, y, prime;

  cgbn_load(po._env, x, &(inputs[instance].x));
  cgbn_load(po._env, y, &(inputs[instance].y));
  cgbn_load(po._env, prime, constants);

  uint32_t np0 = cgbn_bn2mont(po._env, x, x, prime);
  cgbn_bn2mont(po._env, y, y, prime);
  cgbn_mont_mul(po._env, x, x, y, prime, np0);
  cgbn_mont2bn(po._env, x, x, prime, np0);

  cgbn_store(po._env, &(outputs[instance]), x);
}

// Multiply x, y, and z mod prime
template<class params>
__device__ void kernel_mul3(cgbn_error_report_t *report, typename cmixPrecomp<params>::mem_t *constants, typename cmixPrecomp<params>::mul3_input_t *inputs, typename cmixPrecomp<params>::mem_t *outputs, size_t count) {
  int32_t instance;

  // decode an instance number from the blockIdx and threadIdx
  instance=(blockIdx.x*blockDim.x + threadIdx.x)/params::TPI;
  if(instance>=count)
    return;

  cmixPrecomp<params>                 po(cgbn_report_monitor, report, instance);
  typename cmixPrecomp<params>::bn_t  x, y, prime;

  cgbn_load(po._env, x, &(inputs[instance].x));
  cgbn_load(po._env, y, &(inputs[instance].y));
  cgbn_load(po._env, prime, constants);

  // Pattern: result stored in first non env param
  uint32_t np0 = cgbn_bn2mont(po._env, x, x, prime);
  cgbn_bn2mont(po._env, y, y, prime);
  cgbn_mont_mul(po._env, x, x, y, prime, np0);
  cgbn_load(po._env,y , &(inputs[instance].z));
  cgbn_bn2mont(po._env, y, y, prime);
  cgbn_mont_mul(po._env, x, x, y, prime, np0);
  cgbn_mont2bn(po._env, x, x, prime, np0);

  cgbn_store(po._env, &(outputs[instance]), x);
}


// Run powm kernel
// Enqueues kernel on the stream and returns immediately (non-blocking)
// The results will be placed in the stream's gpu outputs buffer some time after the kernel launch
// Precondition: stream should have had upload called on it
// We need to invoke kernels using the cuda driver API, which is a little more complicated
template<class params>
const char* run(streamData *stream) {
  debugPrint("run (streamData, kernel)");
  const int32_t              TPB=(params::TPB==0) ? 128 : params::TPB;    // default threads per block to 128
  const int32_t              TPI=params::TPI, IPB=TPB/TPI;                // IPB is instances per block
  unsigned int gridDimX = (stream->length+IPB-1)/IPB;
  unsigned int blockDimX = TPB;

  CU_CHECK_RETURN(cuStreamWaitEvent(stream->stream, stream->hostToDevice, 0));
  // launch kernel with blocks=ceil(instance_count/IPB) and threads=TPB
  // TODO We should be able to launch more than just this kernel.
  //  Organize with enumeration? Is it possible to use templates to make this better?
  typedef typename cmixPrecomp<params>::mem_t mem_t;

  CUfunction function;
  // Report and number of items are always the same for all calls
  void *kernelParams[5] = { &stream->report, NULL, NULL, NULL, &stream->length };

  // select function and parameters locations
  switch (stream->whichToRun) {
  case KERNEL_POWM_ODD:
    {
      typedef typename cmixPrecomp<params>::powm_odd_input_t input_t;
      mem_t* gpuConstants = (mem_t*)stream->gpuMem;
      input_t* gpuInputs = (input_t*)(gpuConstants+1);
      mem_t* gpuOutputs = (mem_t*)(gpuInputs+stream->length);
      switch (params::BITS) {
        case 2048:
          function = powm_2048_function;
          break;
        case 3200:
          function = powm_3200_function;
          break;
        case 4096:
          function = powm_4096_function;
          break;
      }
      kernelParams[1] = &gpuConstants;
      kernelParams[2] = &gpuInputs;
      kernelParams[3] = &gpuOutputs;
      CU_CHECK_RETURN(cuLaunchKernel(function, gridDimX, 1, 1, blockDimX, 1, 1, 0, stream->stream, kernelParams, NULL));
    }
    break;
  case KERNEL_ELGAMAL:
    {
      typedef typename cmixPrecomp<params>::elgamal_input_t input_t;
      typedef typename cmixPrecomp<params>::elgamal_output_t output_t;
      typedef typename cmixPrecomp<params>::elgamal_constant_t constant_t;
      constant_t* gpuConstants = (constant_t*)stream->gpuMem;
      input_t* gpuInputs = (input_t*)(gpuConstants+1);
      output_t* gpuOutputs = (output_t*)(gpuInputs+stream->length);
      switch (params::BITS) {
        case 2048:
          function = elgamal_2048_function;
          break;
        case 3200:
          function = elgamal_3200_function;
          break;
        case 4096:
          function = elgamal_4096_function;
          break;
      }
      kernelParams[1] = &gpuConstants;
      kernelParams[2] = &gpuInputs;
      kernelParams[3] = &gpuOutputs;
      CU_CHECK_RETURN(cuLaunchKernel(function, gridDimX, 1, 1, blockDimX, 1, 1, 0, stream->stream, kernelParams, NULL));
    }
    break;
  case KERNEL_REVEAL:
    {
      typedef typename cmixPrecomp<params>::reveal_constant_t constant_t;
      constant_t* gpuConstants = (constant_t*)stream->gpuMem;
      mem_t* gpuInputs = (mem_t*)(gpuConstants+1);
      mem_t* gpuOutputs = (mem_t*)(gpuInputs+stream->length);
      switch (params::BITS) {
        case 2048:
          function = reveal_2048_function;
          break;
        case 3200:
          function = reveal_3200_function;
          break;
        case 4096:
          function = reveal_4096_function;
          break;
      }
      kernelParams[1] = &gpuConstants;
      kernelParams[2] = &gpuInputs;
      kernelParams[3] = &gpuOutputs;
      CU_CHECK_RETURN(cuLaunchKernel(function, gridDimX, 1, 1, blockDimX, 1, 1, 0, stream->stream, kernelParams, NULL));
    }
    break;
  case KERNEL_MUL2:
    {
      typedef typename cmixPrecomp<params>::mul2_input_t input_t;
      mem_t *gpuConstants = (mem_t*)(stream->gpuMem);
      input_t *gpuInputs = (input_t*)(gpuConstants+1);
      mem_t *gpuOutputs = (mem_t*)(gpuInputs+stream->length);
      switch (params::BITS) {
        case 2048:
          function = mul2_2048_function;
          break;
        case 3200:
          function = mul2_3200_function;
          break;
        case 4096:
          function = mul2_4096_function;
          break;
      }
      kernelParams[1] = &gpuConstants;
      kernelParams[2] = &gpuInputs;
      kernelParams[3] = &gpuOutputs;
      CU_CHECK_RETURN(cuLaunchKernel(function, gridDimX, 1, 1, blockDimX, 1, 1, 0, stream->stream, kernelParams, NULL));
    }
    break;
  case KERNEL_MUL3:
    {
      typedef typename cmixPrecomp<params>::mul3_input_t input_t;
      mem_t *gpuConstants = (mem_t*)(stream->gpuMem);
      input_t *gpuInputs = (input_t*)(gpuConstants+1);
      mem_t *gpuOutputs = (mem_t*)(gpuInputs+stream->length);
      switch (params::BITS) {
        case 2048:
          function = mul3_2048_function;
          break;
        case 3200:
          function = mul3_3200_function;
          break;
        case 4096:
          function = mul3_4096_function;
          break;
      }
      kernelParams[1] = &gpuConstants;
      kernelParams[2] = &gpuInputs;
      kernelParams[3] = &gpuOutputs;
      CU_CHECK_RETURN(cuLaunchKernel(function, gridDimX, 1, 1, blockDimX, 1, 1, 0, stream->stream, kernelParams, NULL));
    }
    break;
  default:
    return strdup("Unknown kernel not implemented");
    break;
  }

  CU_CHECK_RETURN(cuEventRecord(stream->exec, stream->stream));

  return NULL;
}

const char* getResults(streamData *stream) {
  debugPrint("getResults (streamData)");
  // Wait for download to complete
  CU_CHECK_RETURN(cuEventSynchronize(stream->deviceToHost));
  // Not sure if we can check the error report before this (e.g. in download function)
  // Could CGBN errors potentially not get returned if stream->deviceToHost doesn't get recorded for some reason?
  CGBN_CHECK_RETURN(stream->report);
  return NULL;
}

// Get the location where the outputs start for uploading
//  TODO Should this be located somewhere else? Could I reduce redundancy if I make different classes for each op? (for instance)
//  Basically this design seems suboptimal in some way
template <class Input, class Constant>
void* getOutputs(void* mem, size_t numItems) {
  debugPrint("getOutputs (void*, size_t)");
  // We want to get the location after the inputs and constants
  // Order doesn't matter and we assume padding doesn't exist
  // (which should be true if the big numbers are big enough)
  Constant* constants = (Constant*)mem;
  // Not sure if this is undefined behaviour? Do I have to reinterpret_cast or cast to void to make this work?
  Input* inputs = (Input*)(constants+1);
  return (void*) (inputs+numItems);
}

template <class Input, class Constant>
CUdeviceptr getOutputs(CUdeviceptr mem, size_t numItems) {
  debugPrint("getOutputs (CUdeviceptr, size_t)");
  // We want to get the location after the inputs and constants
  // Order doesn't matter and we assume padding doesn't exist
  // (which should be true if the big numbers are big enough)
  // Does this cast work OK?
  Constant* constants = (Constant*)mem;
  // Not sure if this is undefined behaviour? Do I have to reinterpret_cast or cast to void to make this work?
  Input* inputs = (Input*)(constants+1);
  return (CUdeviceptr) (inputs+numItems);
}

// Get the memory address of the beginning of the inputs in a buffer
// Inputs always come right after constants
template <class Constant>
void* getInputs(void* mem) {
  debugPrint("getInputs (void*, size_t)");
  Constant* constants = (Constant*)mem;
  return (void*)(constants+1);
}


typedef powm_params_t<32, 4096, 5> params4096;
typedef powm_params_t<32, 3200, 5> params3200;
typedef powm_params_t<32, 2048, 5> params2048;

// TODO? Deprecate this and use sizeof / pointer arithmetic instead
//  Although, I think we might need this for slicing things on the go side
// TODO? Support different bit lengths
template<class cgbnParams>
size_t getConstantsSize(enum kernel op) {
  switch (op) {
    case KERNEL_ELGAMAL:
      return sizeof(typename cmixPrecomp<cgbnParams>::elgamal_constant_t);
      break;
    case KERNEL_POWM_ODD:
    case KERNEL_MUL2:
    case KERNEL_MUL3:
      return sizeof(typename cmixPrecomp<cgbnParams>::mem_t);
      break;
    case KERNEL_REVEAL:
      return sizeof(typename cmixPrecomp<cgbnParams>::reveal_constant_t);
      break;
    default:
      // Unimplemented
      return 0;
      break;
  }
}

template<class cgbnParams>
size_t getInputSize(enum kernel op) {
  switch (op) {
    case KERNEL_ELGAMAL:
      return sizeof(typename cmixPrecomp<cgbnParams>::elgamal_input_t);
      break;
    case KERNEL_POWM_ODD:
      return sizeof(typename cmixPrecomp<cgbnParams>::powm_odd_input_t);
      break;
    case KERNEL_REVEAL:
      return sizeof(typename cmixPrecomp<cgbnParams>::mem_t);
      break;
    case KERNEL_MUL2:
      return sizeof(typename cmixPrecomp<cgbnParams>::mul2_input_t);
      break;
    case KERNEL_MUL3:
      return sizeof(typename cmixPrecomp<cgbnParams>::mul3_input_t);
      break;
    default:
      // Unimplemented
      return 0;
      break;
  }
}

template<class cgbnParams>
size_t getOutputSize(enum kernel op) {
  switch (op) {
    case KERNEL_ELGAMAL:
      return sizeof(typename cmixPrecomp<cgbnParams>::elgamal_output_t);
      break;
    case KERNEL_POWM_ODD:
    case KERNEL_REVEAL:
    case KERNEL_MUL2:
    case KERNEL_MUL3:
      // Most ops just return one number
      return sizeof(typename cmixPrecomp<cgbnParams>::mem_t);
      break;
    default:
      // Unimplemented
      return 0;
      break;
  }
}

// Enqueue upload for a specific kernel
// Stage input data by copying to the stream's constants and inputs memory before calling
template<class cgbnParams>
const char* upload(const uint32_t instance_count, void *stream, enum kernel whichToRun) {
  auto gpuData = (streamData*)stream;
  // Previous download must finish before data are uploaded
  CU_CHECK_RETURN(cuStreamWaitEvent(gpuData->stream, gpuData->deviceToHost, 0));
  
  // Set instance count; it's re-used when the kernel gets run later
  gpuData->length = instance_count;
  // It also doesn't take too long to bounds check the requested data transfer sizes
  // to avoid segmentation faults
  // Avoid these errors by querying the stream for how many items of a particular kernel it can run before uploading
  size_t inputsUploadSize = getInputSize<cgbnParams>(whichToRun) * instance_count + getConstantsSize<cgbnParams>(whichToRun);
  size_t outputsDownloadSize = getOutputSize<cgbnParams>(whichToRun) * instance_count;
  if (inputsUploadSize + outputsDownloadSize > gpuData->memCapacity) {
    return strdup("upload error: inputs+outputs larger than stream capacity\n");
  }

  // At this point, everything should be in a good state, so set the needed variables
  gpuData->inputsLength = inputsUploadSize;
  gpuData->outputsLength = outputsDownloadSize;
  gpuData->whichToRun = whichToRun;

  // Record starting time of this upload
  CU_CHECK_RETURN(cuEventRecord(gpuData->start, gpuData->stream));

  // Upload the inputs
  CU_CHECK_RETURN(cuMemcpyHtoDAsync(gpuData->gpuMem, gpuData->cpuMem, gpuData->inputsLength, gpuData->stream));

  // Run should wait on this event for kernel launch
  // The event should include any memcpys that haven't completed yet
  CU_CHECK_RETURN(cuEventRecord(gpuData->hostToDevice, gpuData->stream));

  return NULL;
}

// Trigger a kernel run
template<class cgbnParams>
const char* run(void *stream) {
  return run<cgbnParams>((streamData*)stream);
}


template<class cgbnParams>
const char* download(void *s) {
  auto stream = (streamData*)s;
  // Wait for the kernel to finish running
  CU_CHECK_RETURN(cuStreamWaitEvent(stream->stream, stream->exec, 0));

  // The kernel ran successfully, so we get the results off the GPU
  // Outputs come right after the inputs
  // TODO: Do this differently (need to not do arithmetic on void pointer for it to be valid)
  // Specifically, I know what the byte length should be in this buffer because the operation is specified
  // Basically, we're completing the type manually for bindings compatibility reasons
  // Otherwise we'd have to template(?) the streamData struct with the input, output, and constant types
  // and cast the cpu and gpu buffers to the right structure depending on the operation.
  // 
  // For now, just instantiate with types for elgamal 4k?
  // This is a mess. We should just be able to pass "elgamal" or, at worst, "elgamal<params4096>" 
  // I'd rather not have to switch all the types and instantiate different classes based on the cryptop but that's a limitation on Cgo.
  void *cpuOutputs;
  CUdeviceptr gpuOutputs;
  typedef typename cmixPrecomp<cgbnParams>::mem_t mem_t;
  switch (stream->whichToRun) {
  case KERNEL_ELGAMAL:
    cpuOutputs = getOutputs<typename cmixPrecomp<cgbnParams>::elgamal_input_t, typename cmixPrecomp<cgbnParams>::elgamal_constant_t>(stream->cpuMem, stream->length);
    gpuOutputs = getOutputs<typename cmixPrecomp<cgbnParams>::elgamal_input_t, typename cmixPrecomp<cgbnParams>::elgamal_constant_t>(stream->gpuMem, stream->length);
    break;
  case KERNEL_POWM_ODD:
    cpuOutputs = getOutputs<typename cmixPrecomp<cgbnParams>::powm_odd_input_t, mem_t>(stream->cpuMem, stream->length);
    gpuOutputs = getOutputs<typename cmixPrecomp<cgbnParams>::powm_odd_input_t, mem_t>(stream->gpuMem, stream->length);
    break;
  case KERNEL_REVEAL:
    cpuOutputs = getOutputs<mem_t, typename cmixPrecomp<cgbnParams>::reveal_constant_t>(stream->cpuMem, stream->length);
    gpuOutputs = getOutputs<mem_t, typename cmixPrecomp<cgbnParams>::reveal_constant_t>(stream->gpuMem, stream->length);
    break;
  case KERNEL_MUL2:
    cpuOutputs = getOutputs<typename cmixPrecomp<cgbnParams>::mul2_input_t, mem_t>(stream->cpuMem, stream->length);
    gpuOutputs = getOutputs<typename cmixPrecomp<cgbnParams>::mul2_input_t, mem_t>(stream->gpuMem, stream->length);
    break;
  case KERNEL_MUL3:
    cpuOutputs = getOutputs<typename cmixPrecomp<cgbnParams>::mul3_input_t, mem_t>(stream->cpuMem, stream->length);
    gpuOutputs = getOutputs<typename cmixPrecomp<cgbnParams>::mul3_input_t, mem_t>(stream->gpuMem, stream->length);
    break;
  default:
    return strdup("Unknown kernel for download; unable to find location of outputs in buffer\n");
  }
  CU_CHECK_RETURN(cuMemcpyDtoHAsync(cpuOutputs, gpuOutputs, stream->outputsLength, stream->stream));
  CU_CHECK_RETURN(cuEventRecord(stream->deviceToHost, stream->stream));

  return NULL;
}

// Return cpu inputs buffer pointer for writing
template<class cgbnParams>
void* getCpuInputs(void* stream, enum kernel op) {
  streamData* s = (streamData*)stream;
  switch (op) {
    case KERNEL_ELGAMAL:
      return getInputs<typename cmixPrecomp<cgbnParams>::elgamal_constant_t>(s->cpuMem);
      break;
    case KERNEL_POWM_ODD:
    case KERNEL_MUL2:
    case KERNEL_MUL3:
      return getInputs<typename cmixPrecomp<cgbnParams>::mem_t>(s->cpuMem);
      break;
    case KERNEL_REVEAL:
      return getInputs<typename cmixPrecomp<cgbnParams>::reveal_constant_t>(s->cpuMem);
      break;
    default:
      // Unimplemented
      return NULL;
      break;
  }
}

// Return cpu outputs buffer pointer for reading
template<class cgbnParams>
void* getCpuOutputs(void* stream) {
  streamData* s = (streamData*)stream;
  switch (s->whichToRun) {
    case KERNEL_ELGAMAL:
      return getOutputs<typename cmixPrecomp<cgbnParams>::elgamal_input_t, typename cmixPrecomp<cgbnParams>::elgamal_constant_t>(
          s->cpuMem, s->length);
      break;
    case KERNEL_POWM_ODD:
      return getOutputs<typename cmixPrecomp<cgbnParams>::powm_odd_input_t, typename cmixPrecomp<cgbnParams>::mem_t>(
          s->cpuMem, s->length);
      break;
    case KERNEL_REVEAL:
      return getOutputs<typename cmixPrecomp<cgbnParams>::mem_t, typename cmixPrecomp<cgbnParams>::reveal_constant_t>(
          s->cpuMem, s->length);
      break;
    case KERNEL_MUL2:
      return getOutputs<typename cmixPrecomp<cgbnParams>::mul2_input_t, typename cmixPrecomp<cgbnParams>::mem_t>(
          s->cpuMem, s->length);
      break;
    case KERNEL_MUL3:
      return getOutputs<typename cmixPrecomp<cgbnParams>::mul3_input_t, typename cmixPrecomp<cgbnParams>::mem_t>(
          s->cpuMem, s->length);
      break;
    default:
      // Unimplemented
      return NULL;
      break;
  }
}

// create a bunch of streams and buffers suitable for running a particular kernel
inline const char* createStream(streamCreateInfo createInfo, streamData* stream) {
  CU_CHECK_RETURN(cuCtxPushCurrent(cuContext));
  debugPrint("createStream (streamData)");
  stream->memCapacity = createInfo.capacity;
  stream->length = 0;
  stream->outputsLength = 0;
  stream->inputsLength = 0;
  CU_CHECK_RETURN(cuStreamCreate(&(stream->stream), CU_STREAM_DEFAULT));
  CU_CHECK_RETURN(cuMemAlloc(&(stream->gpuMem), createInfo.capacity));
  CU_CHECK_RETURN(cgbn_error_report_alloc(&stream->report));
  CU_CHECK_RETURN(cuMemAllocHost(&(stream->cpuMem), createInfo.capacity));

  // cudaEventBlockingSync prevents 100% cpu usage when synchronizing on an event
  // However, it may impact performance, so I turned it off for now(?)
  CU_CHECK_RETURN(cuEventCreate(&stream->start, CU_EVENT_DISABLE_TIMING));
  CU_CHECK_RETURN(cuEventCreate(&stream->hostToDevice, CU_EVENT_DISABLE_TIMING));
  CU_CHECK_RETURN(cuEventCreate(&stream->exec, CU_EVENT_DISABLE_TIMING));
  CU_CHECK_RETURN(cuEventCreate(&stream->deviceToHost, CU_EVENT_DISABLE_TIMING));
  CU_CHECK_RETURN(cuCtxPopCurrent(&cuContext));

  return NULL;
}


// All the methods used in cgo should have extern "C" linkage to avoid
// implementation-specific name mangling
// This makes them more straightforward to load from the shared object
extern "C" {
  size_t getConstantsSize4096(enum kernel op) {
    return getConstantsSize<params4096>(op);
  }
  size_t getConstantsSize3200(enum kernel op) {
    return getConstantsSize<params3200>(op);
  }
  size_t getConstantsSize2048(enum kernel op) {
    return getConstantsSize<params2048>(op);
  }
  size_t getInputSize4096(enum kernel op) {
    return getInputSize<params4096>(op);
  }
  size_t getInputSize3200(enum kernel op) {
    return getInputSize<params3200>(op);
  }
  size_t getInputSize2048(enum kernel op) {
    return getInputSize<params2048>(op);
  }
  size_t getOutputSize4096(enum kernel op) {
    return getOutputSize<params4096>(op);
  }
  size_t getOutputSize3200(enum kernel op) {
    return getOutputSize<params3200>(op);
  }
  size_t getOutputSize2048(enum kernel op) {
    return getOutputSize<params2048>(op);
  }
  void* getCpuOutputs4096(void* stream) {
    return getCpuOutputs<params4096>(stream);
  }
  void* getCpuOutputs3200(void* stream) {
    return getCpuOutputs<params3200>(stream);
  }
  void* getCpuOutputs2048(void* stream) {
    return getCpuOutputs<params2048>(stream);
  }

  // Return cpu inputs buffer pointer for writing
  void* getCpuInputs4096(void* stream, enum kernel op) {
    return getCpuInputs<params4096>(stream, op);
  }
  // Return cpu inputs buffer pointer for writing
  void* getCpuInputs3200(void* stream, enum kernel op) {
    return getCpuInputs<params3200>(stream, op);
  }
  // Return cpu inputs buffer pointer for writing
  void* getCpuInputs2048(void* stream, enum kernel op) {
    return getCpuInputs<params2048>(stream, op);
  }

  // Enqueue the specified kernel at 4k bits size
  const char* enqueue4096(const uint32_t instance_count, void *stream, enum kernel whichToRun) {
    CU_CHECK_RETURN(cuCtxPushCurrent(cuContext));
    debugPrint("enqueue4096 (void)");
    const char* err;
    err = upload<params4096>(instance_count, stream, whichToRun);
    if (err != NULL) {
      return err;
    }
    // memset output buffer to make it possible to monitor for results
    err = run<params4096>(stream);
    if (err != NULL) {
      return err;
    }
    CU_CHECK_RETURN(cuCtxPopCurrent(&cuContext));
    return download<params4096>(stream);
  }

  // Enqueue the specified kernel at 3k bits size
  const char* enqueue3200(const uint32_t instance_count, void *stream, enum kernel whichToRun) {
    CU_CHECK_RETURN(cuCtxPushCurrent(cuContext));
    debugPrint("enqueue3200 (void)");
    const char* err;
    err = upload<params3200>(instance_count, stream, whichToRun);
    if (err != NULL) {
      return err;
    }
    err = run<params3200>(stream);
    if (err != NULL) {
      return err;
    }
    CU_CHECK_RETURN(cuCtxPopCurrent(&cuContext));
    return download<params3200>(stream);
  }

  // Enqueue the specified kernel at 2k bits size
  const char* enqueue2048(const uint32_t instance_count, void *stream, enum kernel whichToRun) {
    CU_CHECK_RETURN(cuCtxPushCurrent(cuContext));
    debugPrint("enqueue2048 (void)");
    const char* err;
    err = upload<params2048>(instance_count, stream, whichToRun);
    if (err != NULL) {
      return err;
    }
    err = run<params2048>(stream);
    if (err != NULL) {
      return err;
    }
    CU_CHECK_RETURN(cuCtxPopCurrent(&cuContext));
    return download<params2048>(stream);
  }

  // Returns error and elapsed time of the run in floating-point seconds
  const char* getResults(void *stream) {
    debugPrint("getResults (void)");
    return getResults((streamData*)stream);
  }

  // Call this when starting the program to allocate resources
  // Returns stream or error
  struct stream_return_data* createStream(streamCreateInfo createInfo) {
    debugPrint("createStream (streamCreateInfo)");
    stream_return_data* result = (stream_return_data*)malloc(sizeof(*result));
    if (result == NULL) {
      // Allocation failed, rest of method's not valid
      return NULL;
    }
    streamData *s = (streamData*)(malloc(sizeof(*s)));
    if (s == NULL) {
      // Allocation failed, rest of method's not valid
      // Don't return error string as its allocation is likely to also fail
      free((void*)result);
      return NULL;
    }
    result->error = createStream(createInfo, s);
    result->result = s;
    result->cpuBuf = s->cpuMem;
    return result;
  }

  // Call this after execution has completed to deallocate resources
  // Returns error
  const char* destroyStream(void *destroyee) {
    debugPrint("destroyStream (void)");
    auto stream = (streamData*)destroyee;
    // Don't know at what point there could have been errors while creating this stream,
    // so make sure things exist before destroying them
    CU_CHECK_RETURN(cuCtxPushCurrent(cuContext));
    if (stream != NULL) {
      if (stream->gpuMem != 0) {
        CU_CHECK_RETURN(cuMemFree(stream->gpuMem));
        // Setting to null prevents double free if destroyStream is called twice for some reason
        // This shouldn't happen
        stream->gpuMem = 0;
      }
      if (stream->cpuMem != NULL) {
        CU_CHECK_RETURN(cuMemFreeHost(stream->cpuMem));
        stream->cpuMem = NULL;
      }
      if (stream->report != NULL) {
        CU_CHECK_RETURN(cgbn_error_report_free(stream->report));
        stream->report = NULL;
      }
      if (stream->start != NULL) {
        CU_CHECK_RETURN(cuEventDestroy(stream->start));
        stream->start = NULL;
      }
      if (stream->hostToDevice != NULL) {
        CU_CHECK_RETURN(cuEventDestroy(stream->hostToDevice));
        stream->hostToDevice = NULL;
      }
      if (stream->exec != NULL) {
        CU_CHECK_RETURN(cuEventDestroy(stream->exec));
        stream->exec = NULL;
      }
      if (stream->deviceToHost != NULL) {
        CU_CHECK_RETURN(cuEventDestroy(stream->deviceToHost));
        stream->deviceToHost = NULL;
      }
      if (stream->stream != NULL) {
        CU_CHECK_RETURN(cuStreamDestroy(stream->stream));
        stream->stream = NULL;
      }
      free((void*)stream);
    }
    CU_CHECK_RETURN(cuCtxPopCurrent(&cuContext));

    return NULL;
  }

#define FALSE 0
#define TRUE 1
  // Return true if passed stream has all fields that should be populated after
  // initialization populated
  int isStreamValid(void* stream) {
    if (stream == NULL) {
      return FALSE;
    }
    auto s = (streamData*)stream;
    if (s->stream == NULL) {
      return FALSE;
    }
    if (s->gpuMem == 0) {
      return FALSE;
    }
    if (s->cpuMem == NULL) {
      return FALSE;
    }
    if (s->report == NULL) {
      return FALSE;
    }
    if (s->start == NULL) {
      return FALSE;
    }
    if (s->hostToDevice == NULL) {
      return FALSE;
    }
    if (s->exec == NULL) {
      return FALSE;
    }
    if (s->deviceToHost == NULL) {
      return FALSE;
    }
    if (s->memCapacity == 0) {
      return FALSE;
    }
    return TRUE;
  }

  // Return cpu constants buffer pointer for writing
  void* getCpuConstants(void* stream) {
    return ((streamData*)stream)->cpuMem;
  }

  // Eventually: one context per device?
  // You must call this before any other calls that call cuda
  const char* initCuda() {
    if (cuContext == NULL) {
      CU_CHECK_RETURN(cuInit(0));
      // count available devices
      int deviceCount = 0;
      CU_CHECK_RETURN(cuDeviceGetCount(&deviceCount));
      
      if (deviceCount == 0) {
        return strdup("no CUDA devices available. are drivers installed? try running node with useGPU:false");
      }

      // try first device, which should work if we made it here
      // TODO implement explicit device selection
      int dev = 0;
      int cuDevice = 0;
      CU_CHECK_RETURN(cuDeviceGet(&cuDevice, dev));

      // initialize context on the device we chose
      CU_CHECK_RETURN(cuCtxCreate(&cuContext, 0, cuDevice));
      // Samples have a global cuContext, so that's probably fine for now...
      CU_CHECK_RETURN(cuModuleLoad(&cuModule, "/opt/xxnetwork/lib/libpow.fatbin"));
      CU_CHECK_RETURN(cuModuleGetFunction(&mul2_4096_function, cuModule, "mul2_4096"));
      CU_CHECK_RETURN(cuModuleGetFunction(&mul3_4096_function, cuModule, "mul3_4096"));
      CU_CHECK_RETURN(cuModuleGetFunction(&elgamal_4096_function, cuModule, "elgamal_4096"));
      CU_CHECK_RETURN(cuModuleGetFunction(&reveal_4096_function, cuModule, "reveal_4096"));
      CU_CHECK_RETURN(cuModuleGetFunction(&powm_4096_function, cuModule, "powm_4096"));
      CU_CHECK_RETURN(cuModuleGetFunction(&mul2_3200_function, cuModule, "mul2_3200"));
      CU_CHECK_RETURN(cuModuleGetFunction(&mul3_3200_function, cuModule, "mul3_3200"));
      CU_CHECK_RETURN(cuModuleGetFunction(&elgamal_3200_function, cuModule, "elgamal_3200"));
      CU_CHECK_RETURN(cuModuleGetFunction(&reveal_3200_function, cuModule, "reveal_3200"));
      CU_CHECK_RETURN(cuModuleGetFunction(&powm_3200_function, cuModule, "powm_3200"));
      CU_CHECK_RETURN(cuModuleGetFunction(&mul2_2048_function, cuModule, "mul2_2048"));
      CU_CHECK_RETURN(cuModuleGetFunction(&mul3_2048_function, cuModule, "mul3_2048"));
      CU_CHECK_RETURN(cuModuleGetFunction(&elgamal_2048_function, cuModule, "elgamal_2048"));
      CU_CHECK_RETURN(cuModuleGetFunction(&reveal_2048_function, cuModule, "reveal_2048"));
      CU_CHECK_RETURN(cuModuleGetFunction(&powm_2048_function, cuModule, "powm_2048"));
      return NULL;
    }
    return NULL;
  }


}

// Kernel wrappers with "C"
// These are needed to be able to load kernels without having to copy mangled names from cuobjdump
extern "C" {
  __global__ void mul2_4096(cgbn_error_report_t *report, typename cmixPrecomp<params4096>::mem_t *constants, typename cmixPrecomp<params4096>::mul2_input_t *inputs, typename cmixPrecomp<params4096>::mem_t *outputs, size_t count) {
    kernel_mul2<params4096>(report, constants, inputs, outputs, count);
  }
  __global__ void mul2_3200(cgbn_error_report_t *report, typename cmixPrecomp<params3200>::mem_t *constants, typename cmixPrecomp<params3200>::mul2_input_t *inputs, typename cmixPrecomp<params3200>::mem_t *outputs, size_t count) {
    kernel_mul2<params3200>(report, constants, inputs, outputs, count);
  }
  __global__ void mul2_2048(cgbn_error_report_t *report, typename cmixPrecomp<params2048>::mem_t *constants, typename cmixPrecomp<params2048>::mul2_input_t *inputs, typename cmixPrecomp<params2048>::mem_t *outputs, size_t count) {
    kernel_mul2<params2048>(report, constants, inputs, outputs, count);
  }
__global__ void mul3_4096(cgbn_error_report_t *report, typename cmixPrecomp<params4096>::mem_t *constants, typename cmixPrecomp<params4096>::mul3_input_t *inputs, typename cmixPrecomp<params4096>::mem_t *outputs, size_t count) {
    kernel_mul3<params4096>(report, constants, inputs, outputs, count);
  }
__global__ void mul3_3200(cgbn_error_report_t *report, typename cmixPrecomp<params3200>::mem_t *constants, typename cmixPrecomp<params3200>::mul3_input_t *inputs, typename cmixPrecomp<params3200>::mem_t *outputs, size_t count) {
    kernel_mul3<params3200>(report, constants, inputs, outputs, count);
  }
__global__ void mul3_2048(cgbn_error_report_t *report, typename cmixPrecomp<params2048>::mem_t *constants, typename cmixPrecomp<params2048>::mul3_input_t *inputs, typename cmixPrecomp<params2048>::mem_t *outputs, size_t count) {
    kernel_mul3<params2048>(report, constants, inputs, outputs, count);
  }
  __global__ void powm_4096(cgbn_error_report_t *report, typename cmixPrecomp<params4096>::mem_t *constants, typename cmixPrecomp<params4096>::powm_odd_input_t *inputs, typename cmixPrecomp<params4096>::mem_t *outputs, size_t count) {
    kernel_powm_odd<params4096>(report, constants, inputs, outputs, count);
  }
  __global__ void powm_3200(cgbn_error_report_t *report, typename cmixPrecomp<params3200>::mem_t *constants, typename cmixPrecomp<params3200>::powm_odd_input_t *inputs, typename cmixPrecomp<params3200>::mem_t *outputs, size_t count) {
    kernel_powm_odd<params3200>(report, constants, inputs, outputs, count);
  }
  __global__ void powm_2048(cgbn_error_report_t *report, typename cmixPrecomp<params2048>::mem_t *constants, typename cmixPrecomp<params2048>::powm_odd_input_t *inputs, typename cmixPrecomp<params2048>::mem_t *outputs, size_t count) {
    kernel_powm_odd<params2048>(report, constants, inputs, outputs, count);
  }
  __global__ void elgamal_4096(cgbn_error_report_t *report, typename cmixPrecomp<params4096>::elgamal_constant_t *constants, typename cmixPrecomp<params4096>::elgamal_input_t *inputs, typename cmixPrecomp<params4096>::elgamal_output_t *outputs, size_t count) {
    kernel_elgamal<params4096>(report, constants, inputs, outputs, count);
  }
  __global__ void elgamal_3200(cgbn_error_report_t *report, typename cmixPrecomp<params3200>::elgamal_constant_t *constants, typename cmixPrecomp<params3200>::elgamal_input_t *inputs, typename cmixPrecomp<params3200>::elgamal_output_t *outputs, size_t count) {
    kernel_elgamal<params3200>(report, constants, inputs, outputs, count);
  }
  __global__ void elgamal_2048(cgbn_error_report_t *report, typename cmixPrecomp<params2048>::elgamal_constant_t *constants, typename cmixPrecomp<params2048>::elgamal_input_t *inputs, typename cmixPrecomp<params2048>::elgamal_output_t *outputs, size_t count) {
    kernel_elgamal<params2048>(report, constants, inputs, outputs, count);
  }
  __global__ void reveal_4096(cgbn_error_report_t *report, typename cmixPrecomp<params4096>::reveal_constant_t *constants, typename cmixPrecomp<params4096>::mem_t *inputs, typename cmixPrecomp<params4096>::mem_t *outputs, size_t count) {
    kernel_reveal<params4096>(report, constants, inputs, outputs, count);
  }
  __global__ void reveal_3200(cgbn_error_report_t *report, typename cmixPrecomp<params3200>::reveal_constant_t *constants, typename cmixPrecomp<params3200>::mem_t *inputs, typename cmixPrecomp<params3200>::mem_t *outputs, size_t count) {
    kernel_reveal<params3200>(report, constants, inputs, outputs, count);
  }
  __global__ void reveal_2048(cgbn_error_report_t *report, typename cmixPrecomp<params2048>::reveal_constant_t *constants, typename cmixPrecomp<params2048>::mem_t *inputs, typename cmixPrecomp<params2048>::mem_t *outputs, size_t count) {
    kernel_reveal<params2048>(report, constants, inputs, outputs, count);
  }
}

