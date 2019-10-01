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
struct kernel_return {
  void *results;
  const char *error;
};
// 2K bits
struct kernel_return* powm_2048(const void *prime, const void *instances, const uint32_t instance_count);
// 4K bits
struct kernel_return* powm_4096(const void *prime, const void *instances, const uint32_t instance_count);

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
