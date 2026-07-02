#include "kernel_operator.h"

using namespace AscendC;

namespace ComplexTransposeKernel {
constexpr uint32_t ROWS = 136;
constexpr uint32_t INNER = 8;
constexpr uint32_t COLS = 256;
constexpr uint32_t PLANE_ELEMS = INNER * COLS;
constexpr uint32_t PLANE_BYTES = PLANE_ELEMS * sizeof(float);
constexpr uint32_t INNER_BYTES = INNER * sizeof(float);

__aicore__ inline uint64_t GetAivBlockNum()
{
    return static_cast<uint64_t>(GetBlockNum()) * static_cast<uint64_t>(GetTaskRation());
}

__aicore__ inline void FillOutputVector(LocalTensor<float> &outRLocal,
                                        LocalTensor<float> &outILocal,
                                        LocalTensor<float> &inRLocal,
                                        LocalTensor<float> &inILocal,
                                        uint32_t col)
{
    for (uint32_t inner = 0; inner < INNER; ++inner) {
        const uint32_t srcOffset = inner * COLS + col;
        outRLocal.SetValue(inner, inRLocal.GetValue(srcOffset));
        outILocal.SetValue(inner, inILocal.GetValue(srcOffset));
    }
}

__aicore__ inline void ProcessPlane(GlobalTensor<float> &arGm,
                                    GlobalTensor<float> &aiGm,
                                    GlobalTensor<float> &crGm,
                                    GlobalTensor<float> &ciGm,
                                    TBuf<> &inRBuf,
                                    TBuf<> &inIBuf,
                                    TBuf<> &outRBuf,
                                    TBuf<> &outIBuf,
                                    uint32_t plane,
                                    uint32_t lastDim)
{
    LocalTensor<float> inRLocal = inRBuf.Get<float>();
    LocalTensor<float> inILocal = inIBuf.Get<float>();
    LocalTensor<float> outRLocal = outRBuf.Get<float>();
    LocalTensor<float> outILocal = outIBuf.Get<float>();

    const uint32_t batchIdx = plane / ROWS;
    const uint32_t row = plane - batchIdx * ROWS;
    const uint64_t inputOffset = static_cast<uint64_t>(plane) * PLANE_ELEMS;
    DataCopy(inRLocal, arGm[inputOffset], PLANE_ELEMS);
    DataCopy(inILocal, aiGm[inputOffset], PLANE_ELEMS);
    PipeBarrier<PIPE_ALL>();

    const uint64_t rowOutputBase = static_cast<uint64_t>(row) * COLS * lastDim;
    const uint32_t batchOutputOffset = batchIdx * INNER;
    for (uint32_t col = 0; col < COLS; ++col) {
        FillOutputVector(outRLocal, outILocal, inRLocal, inILocal, col);
        PipeBarrier<PIPE_ALL>();

        const uint64_t outputOffset = rowOutputBase +
                                      static_cast<uint64_t>(col) * lastDim +
                                      batchOutputOffset;
        DataCopy(crGm[outputOffset], outRLocal, INNER);
        DataCopy(ciGm[outputOffset], outILocal, INNER);
        PipeBarrier<PIPE_ALL>();
    }
}
} // namespace ComplexTransposeKernel

extern "C" __global__ __aicore__ void complex_transpose(GM_ADDR a_r,
                                                         GM_ADDR a_i,
                                                         GM_ADDR c_r,
                                                         GM_ADDR c_i,
                                                         GM_ADDR workspace,
                                                         GM_ADDR tiling)
{
#if defined(__DAV_CUBE__)
    (void)a_r;
    (void)a_i;
    (void)c_r;
    (void)c_i;
    (void)workspace;
    (void)tiling;
    return;
#else
    (void)workspace;
    GET_TILING_DATA(tilingData, tiling);
    using namespace ComplexTransposeKernel;

    const uint32_t totalPlanes = tilingData.totalPlanes;
    const uint32_t lastDim = tilingData.lastDim;
    if (totalPlanes == 0 || lastDim == 0) {
        return;
    }

    GlobalTensor<float> arGm;
    GlobalTensor<float> aiGm;
    GlobalTensor<float> crGm;
    GlobalTensor<float> ciGm;
    const uint64_t inputElems = static_cast<uint64_t>(totalPlanes) * PLANE_ELEMS;
    const uint64_t outputElems = static_cast<uint64_t>(ROWS) * COLS * lastDim;
    arGm.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(a_r), inputElems);
    aiGm.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(a_i), inputElems);
    crGm.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(c_r), outputElems);
    ciGm.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(c_i), outputElems);

    TPipe pipe;
    TBuf<> inRBuf;
    TBuf<> inIBuf;
    TBuf<> outRBuf;
    TBuf<> outIBuf;
    pipe.InitBuffer(inRBuf, PLANE_BYTES);
    pipe.InitBuffer(inIBuf, PLANE_BYTES);
    pipe.InitBuffer(outRBuf, INNER_BYTES);
    pipe.InitBuffer(outIBuf, INNER_BYTES);

    const uint64_t blockIdx = static_cast<uint64_t>(GetBlockIdx());
    const uint64_t blockNum = GetAivBlockNum();
    for (uint64_t plane = blockIdx; plane < totalPlanes; plane += blockNum) {
        ProcessPlane(arGm, aiGm, crGm, ciGm,
                     inRBuf, inIBuf, outRBuf, outIBuf,
                     static_cast<uint32_t>(plane), lastDim);
    }
#endif
}
