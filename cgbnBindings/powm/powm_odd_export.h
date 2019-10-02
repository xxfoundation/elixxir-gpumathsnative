// This + linking the so should allow Go to use the bindings I've made more easily.
//
// We are only going to export the "extern C" linked methods, so let's just put those in the header to start...
// Nothing with a template can go in this header.

#ifndef POWM_ODD_EXPORT_H
#define POWM_ODD_EXPORT_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif // __cplusplus

// This type works well when returning large chunks of data to Go
// It's used for the exported bindings for kernel running and data upload
struct return_data {
  // It's difficult for Golang to interpret most of this data, so it should
  // be treated as read-only when the results are used, and unused otherwise.
  // For instance, the upload result gets passed back to the kernel run 
  // method, and it shouldn't be modified or the kernel won't run correctly.
  void *result;
  // Go should check and handle this like a normal Go error - return, handle,
  // or panic.
  const char *error;
};

// Upload data for a powm kernel run for 4K bits
struct return_data* upload_powm_4096(const void *prime, const void *instances, const uint32_t instance_count);
// Run powm for 4K bits
struct return_data* run_powm_4096(const void *prime, const void *instances, const uint32_t instance_count);

// Call this after execution has completed to write out profile information to the disk
const char* stopProfiling();

// Calling this is optional if you profile from the start of execution.
const char* startProfiling();

// If using the newer profiler, use this instead when kernels have finished 
// running to signal the profiler that execution has finished.
const char* resetDevice();
#ifdef __cplusplus
}
#endif // __cplusplus
#endif // POWM_ODD_EXPORT_H

