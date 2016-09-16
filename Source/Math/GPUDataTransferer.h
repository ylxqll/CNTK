#pragma once

#ifndef CPUONLY
#include <cuda_runtime_api.h>
#include <cuda.h>
#endif // !CPUONLY

#include "Basics.h"
#include "DataTransferer.h"

#ifdef _WIN32
#ifndef MATH_API
#ifdef MATH_EXPORTS
#define MATH_API __declspec(dllexport)
#else
#define MATH_API __declspec(dllimport)
#endif
#endif /* MATH_API */
#else  // no DLLs in Linux
#define MATH_API
#endif

namespace Microsoft { namespace MSR { namespace CNTK {

class GranularGPUDataTransferer : public DataTransferer
{
public:
    GranularGPUDataTransferer(int deviceId, bool blocking = false);
    ~GranularGPUDataTransferer();

    void CopyGPUToCPUAsync(const void* gpuBuffer, size_t numElements, size_t elementSize, void* cpuBuffer) override;
    void RecordGPUToCPUCopy() override;
    void WaitForCopyGPUToCPU() override;

    void CopyCPUToGPUAsync(const void* cpuBuffer, size_t numElements, size_t elementSize, void* gpuBuffer) override;
    void RecordCPUToGPUCopy() override;
    void WaitForCopyCPUToGPU() override;

protected:
#ifndef CPUONLY
    cudaStream_t m_fetchStream;
    cudaStream_t m_assignStream;

    mutable cudaEvent_t m_fetchCompleteEvent;
    mutable cudaEvent_t m_assignCompleteEvent;
#endif // !CPUONLY

    int m_deviceId;

    // Disallow copy and move construction and assignment
    DISABLE_COPY_AND_MOVE(GranularGPUDataTransferer);

    template <class ElemType>
    friend class GPUDataTransferer;
};

template <class ElemType>
class MATH_API GPUDataTransferer
{
    // Have to have a raw pointer in order not to expose over dll boundary.
    GranularGPUDataTransferer* m_inner;

public:
    GPUDataTransferer(int deviceId, bool useConcurrentStreams);
    ~GPUDataTransferer();

    // Disallow copy and move construction and assignment
    DISABLE_COPY_AND_MOVE(GPUDataTransferer);

    void CopyGPUToCPUAsync(ElemType* gpuBuffer, size_t numElements, ElemType* cpuBuffer);
    void CopyCPUToGPUAsync(ElemType* cpuBuffer, size_t numElements, ElemType* gpuBuffer);

    void WaitForCopyGPUToCPUAsync();
    void WaitForCopyCPUToGPUAsync();

#ifndef CPUONLY
    static cudaStream_t GetFetchStream();
#endif // !CPUONLY

private:
#ifndef CPUONLY
    static void SyncEvent(cudaEvent_t ev);

    static cudaStream_t s_fetchStream;
    static cudaStream_t s_assignStream;
#endif // !CPUONLY
};

class PrefetchGPUDataTransferer : public GranularGPUDataTransferer
{
public:
    PrefetchGPUDataTransferer(int deviceId);

private:
#ifndef CPUONLY
    static cudaStream_t s_prefetchStream;
#endif

    DISABLE_COPY_AND_MOVE(PrefetchGPUDataTransferer);
};

}}}
