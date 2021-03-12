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

#include <sstream>

// support routines
// The caller is responsible for freeing the returned string
const char* cuda_check(cudaError_t status, const char *action=NULL, const char *file=NULL, int32_t line=0) {
  // check for cuda errors

  if(status!=cudaSuccess) {
    std::stringstream errorMsg;
    errorMsg << "CUDA error occurred: " << cudaGetErrorString(status) << "\n";
    if(action!=NULL) {
      errorMsg << "While running " << action << "   (file " << file << ", line " << line << ")" << std::endl;
    }
    return strdup(errorMsg.str().c_str());
  }
  return NULL;
}

// The caller is responsible for freeing the returned string
// Note: the allocation can fail and no string will be returned!
const char* cu_check(CUresult status, const char *action=NULL, const char *file=NULL, int32_t line=0) {
  // check for errors, driver API edition

  if(status!=CUDA_SUCCESS) {
    std::stringstream errorMsg;
    const char* errName;
    cuGetErrorName(status, &errName);
    errorMsg << "CU error occurred: " << errName << "\n";
    if(action!=NULL) {
      errorMsg << "While running " << action << "   (file " << file << ", line " << line << ")" << std::endl;
    }
    return strdup(errorMsg.str().c_str());
  }
  return NULL;
}

// The caller is responsible for freeing the returned string
const char* cgbn_check(cgbn_error_report_t *report, const char *file=NULL, int32_t line=0) {
  // check for cgbn errors

  if(cgbn_error_report_check(report)) {
    std::stringstream errorMsg;
    errorMsg << "CGBN error occurred: " << cgbn_error_string(report) << "\n";

    if(report->_instance!=0xFFFFFFFF) {
      errorMsg << "Error reported by instance " << report->_instance;
      if(report->_blockIdx.x!=0xFFFFFFFF || report->_threadIdx.x!=0xFFFFFFFF)
        errorMsg << ", ";
      if(report->_blockIdx.x!=0xFFFFFFFF)
        errorMsg << "blockIdx=(" << report->_blockIdx.x << ", " << report->_blockIdx.y <<  ", " <<report->_blockIdx.z << ") ";
      if(report->_threadIdx.x!=0xFFFFFFFF)
        errorMsg << "threadIdx=(" << report->_threadIdx.x << ", " << report->_threadIdx.y << ", " << report->_threadIdx.z << ")";
      errorMsg << std::endl;
    }
    else {
      errorMsg << "Error reported by blockIdx=(" << report->_blockIdx.x << " " << report->_blockIdx.y << " " << report->_blockIdx.z << ")";
      errorMsg << "threadIdx=(" << report->_threadIdx.x << " " << report->_threadIdx.y << " " << report->_threadIdx.z << ")" << std::endl;
    }
    if(file!=NULL)
      errorMsg << "file " << file << ", line " << line << std::endl;
    return strdup(errorMsg.str().c_str());
  }
  return NULL;
}

#define CUDA_CHECK(action) cuda_check(action, #action, __FILE__, __LINE__)
#define CU_CHECK(action) cu_check(action, #action, __FILE__, __LINE__)
#define CGBN_CHECK(report) cgbn_check(report, __FILE__, __LINE__)
// Store the returned error message in a variable, rather than wrapping CUDA_CHECK and CGBN_CHECK calls directly
// Directly wrapping will cause the CHECK calls to be called twice
#define RETURN_IF_EXISTS(errorMsg) if (errorMsg != NULL) return errorMsg
#define PRINT_IF_EXISTS(errorMsg) if (errorMsg != NULL) { printf("%s", errorMsg); free((void*)errorMsg); exit(1); }

#define CUDA_CHECK_RETURN(action) do { \
  const char *err = CUDA_CHECK(action); \
  RETURN_IF_EXISTS(err); \
} while (0);

#define CUDA_CHECK_PRINT(action) do { \
  const char *err = CUDA_CHECK(action); \
  PRINT_IF_EXISTS(err); \
} while (0);

#define CU_CHECK_RETURN(action) do { \
  const char *err = CU_CHECK(action); \
  RETURN_IF_EXISTS(err); \
} while (0);

#define CU_CHECK_PRINT(action) do { \
  const char *err = CU_CHECK(action); \
  PRINT_IF_EXISTS(err); \
} while (0);

#define CGBN_CHECK_RETURN(action) do { \
  const char *err = CGBN_CHECK(action); \
  RETURN_IF_EXISTS(err); \
} while (0);

#define CGBN_CHECK_PRINT(action) do { \
  const char *err = CGBN_CHECK(action); \
  PRINT_IF_EXISTS(err); \
} while (0);
