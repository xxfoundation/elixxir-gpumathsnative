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
const char* upload(const uint32_t instance_count, void *stream, enum kernel whichToRun);
// Enqueue a kernel run
const char* run(void *stream);
// Enqueue download from a previous kernel launch
const char* download(void *stream);
// Wait for a results download to finish
const char* getResults(void *stream);

struct streamCreateInfo {
  // How much memory is available for the stream to use?
  size_t capacity;
};


// Call this when starting the program to allocate resources
// Returns pointer to stream and error
struct return_data* createStream(struct streamCreateInfo createInfo);
// Call this after you're done with the kernel to destroy resources
// Returns error
const char* destroyStream(void *destroyee);

// Get a pointer to the CPU inputs buffer from a stream
// Overwrite this memory with inputs before enqueueing an upload
void* getCpuInputs(void* stream, enum kernel op);

// Get a pointer to the CPU outputs buffer from a stream
// Read outputs from this memory after calling getResults to synchronize the event
void* getCpuOutputs(void* stream);

// Get a pointer to the CPU constants buffer from a stream
// Overwrite this memory with constants before enqueueing an upload
void* getCpuConstants(void* stream);

// Get memory size required for a certain op's constants buffer
size_t getConstantsSize(enum kernel op);
// Get memory size required for a certain op's inputs buffer
size_t getInputSize(enum kernel op);
// Get memory size required for a certain op's outputs buffer
size_t getOutputSize(enum kernel op);

// If using the newer profiler, use this instead when kernels have finished 
// running to signal the profiler that execution has finished.
const char* resetDevice();
#ifdef __cplusplus
}
#endif // __cplusplus
#endif // POWM_ODD_EXPORT_H

