// This + linking the so should allow Go to use the bindings I've made more easily.
//
// We are only going to export the "extern C" linked methods, so let's just put those in the header to start...
// Nothing with a template should go in this header.

#ifndef POWM_ODD_EXPORT_H
#define POWM_ODD_EXPORT_H

#include <stdint.h>
#include <stddef.h>

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

enum kernel {
  KERNEL_POWM_ODD,
  KERNEL_ELGAMAL,
  KERNEL_MUL2,
};

// Prepare a kernel run
const char* upload(const uint32_t instance_count, void *stream, size_t inputsUploadSize, size_t constantsUploadSize, size_t outputsDownloadSize);
// Enqueue a kernel run
const char* run(void *stream, enum kernel whichToRun);
// Enqueue download from a previous kernel launch
const char* download(void *stream);
// Wait for a results download to finish
struct return_data* getResults(void *stream);

struct streamCreateInfo {
  // How many instances can be invoked in a kernel launch?
  size_t capacity;
  // What's the size in bytes of the entire input buffer?
  // (assumed to be linear in size with number of inputs)
  size_t inputsCapacity;
  // What's the size in bytes of the entire output buffer?
  // (assumed to be linear in size with number of inputs)
  size_t outputsCapacity;
  // What's the size in bytes of the entire constants buffer?
  size_t constantsCapacity;
};


// Call this when starting the program to allocate resources
// Returns pointer to stream and error
struct return_data* createStream(struct streamCreateInfo createInfo);
// Call this after you're done with the kernel to destroy resources
// Returns error
const char* destroyStream(void *destroyee);

// Get a pointer to the CPU inputs buffer from a stream
// Overwrite this memory with inputs before enqueueing an upload
void* getCpuInputs(void* stream);

// Get a pointer to the CPU outputs buffer from a stream
// Read outputs from this memory after calling getResults to synchronize the event
void* getCpuOutputs(void* stream);

// Get a pointer to the CPU constants buffer from a stream
// Overwrite this memory with constants before enqueueing an upload
void* getCpuConstants(void* stream);

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

