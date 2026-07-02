#include "kernel_operator.h"

using namespace AscendC;

namespace ComplexTransposeKernel {
constexpr uint32_t ROWS = 136;
constexpr uint32_t INNER = 8;
constexpr uint32_t COLS = 256;
constexpr uint32_t PLANE_ELEMS = INNER * COLS;
constexpr uint32_t BATCH_COPY_CHUNK = 64;
constexpr uint32_t LAST_DIM_COPY_CHUNK = INNER * BATCH_COPY_CHUNK;
constexpr uint32_t LOCAL_COPY_BYTES = LAST_DIM_COPY_CHUNK * sizeof(float);

__aicore__ inline uint64_t GetAivBlockNum()
{
    return static_cast<uint64_t>(GetBlockNum()) * static_cast<uint64_t>(GetTaskRation());
}

__aicore__ inline void FillBatchChunk(GlobalTensor<float> &src,
                                      LocalTensor<float> &dst,
                                      uint32_t row,
                                      uint32_t inner,
                                      uint32_t col,
                                      uint32_t batchStart,
                                      uint32_t len)
{
    for (uint32_t idx = 0; idx < len; ++idx) {
        const uint32_t batchIdx = batchStart + idx;
        const uint64_t srcOffset = (static_cast<uint64_t>(batchIdx) * ROWS + row) * PLANE_ELEMS +
                                   static_cast<uint64_t>(inner) * COLS + col;
        dst.SetValue(idx, src.GetValue(srcOffset));
    }
}

__aicore__ inline void FillLastDimBlock(GlobalTensor<float> &src,
                                        LocalTensor<float> &dst,
                                        uint32_t row,
                                        uint32_t col,
                                        uint32_t batch)
{
    for (uint32_t inner = 0; inner < INNER; ++inner) {
        for (uint32_t batchIdx = 0; batchIdx < batch; ++batchIdx) {
            const uint64_t srcOffset = (static_cast<uint64_t>(batchIdx) * ROWS + row) * PLANE_ELEMS +
                                       static_cast<uint64_t>(inner) * COLS + col;
            const uint32_t dstOffset = inner * batch + batchIdx;
            dst.SetValue(dstOffset, src.GetValue(srcOffset));
        }
    }
}

__aicore__ inline void CopyLocal(GlobalTensor<float> &dst,
                                 uint64_t dstOffset,
                                 LocalTensor<float> &src,
                                 uint32_t len)
{
    DataCopyExtParams copyParams{1, static_cast<uint32_t>(len * sizeof(float)), 0, 0, 0};
    DataCopyPad(dst[dstOffset], src, copyParams);
}

__aicore__ inline void ProcessOutputVector(GlobalTensor<float> &arGm,
                                           GlobalTensor<float> &aiGm,
                                           GlobalTensor<float> &crGm,
                                           GlobalTensor<float> &ciGm,
                                           TBuf<> &outRBuf,
                                           TBuf<> &outIBuf,
                                           uint32_t task,
                                           uint32_t batch,
                                           uint32_t lastDim)
{
    LocalTensor<float> outRLocal = outRBuf.Get<float>();
    LocalTensor<float> outILocal = outIBuf.Get<float>();

    const uint32_t row = task / COLS;
    const uint32_t col = task - row * COLS;
    const uint64_t outputBase = (static_cast<uint64_t>(row) * COLS + col) * lastDim;

    if (batch <= BATCH_COPY_CHUNK) {
        FillLastDimBlock(arGm, outRLocal, row, col, batch);
        FillLastDimBlock(aiGm, outILocal, row, col, batch);
        PipeBarrier<PIPE_ALL>();

        CopyLocal(crGm, outputBase, outRLocal, lastDim);
        CopyLocal(ciGm, outputBase, outILocal, lastDim);
        PipeBarrier<PIPE_ALL>();
        return;
    }

    for (uint32_t inner = 0; inner < INNER; ++inner) {
        const uint64_t innerOutputBase = outputBase + static_cast<uint64_t>(inner) * batch;
        for (uint32_t batchStart = 0; batchStart < batch; batchStart += BATCH_COPY_CHUNK) {
            const uint32_t len = (batchStart + BATCH_COPY_CHUNK <= batch) ? BATCH_COPY_CHUNK : (batch - batchStart);
            FillBatchChunk(arGm, outRLocal, row, inner, col, batchStart, len);
            FillBatchChunk(aiGm, outILocal, row, inner, col, batchStart, len);
            PipeBarrier<PIPE_ALL>();

            CopyLocal(crGm, innerOutputBase + batchStart, outRLocal, len);
            CopyLocal(ciGm, innerOutputBase + batchStart, outILocal, len);
            PipeBarrier<PIPE_ALL>();
        }
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
    const uint32_t batch = tilingData.batch;
    const uint32_t lastDim = tilingData.lastDim;
    if (totalPlanes == 0 || batch == 0 || lastDim == 0) {
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
    TBuf<> outRBuf;
    TBuf<> outIBuf;
    pipe.InitBuffer(outRBuf, LOCAL_COPY_BYTES);
    pipe.InitBuffer(outIBuf, LOCAL_COPY_BYTES);

    const uint64_t blockIdx = static_cast<uint64_t>(GetBlockIdx());
    const uint64_t blockNum = GetAivBlockNum();
    const uint64_t totalTasks = static_cast<uint64_t>(ROWS) * COLS;
    for (uint64_t task = blockIdx; task < totalTasks; task += blockNum) {
        ProcessOutputVector(arGm, aiGm, crGm, ciGm, outRBuf, outIBuf,
                            static_cast<uint32_t>(task), batch, lastDim);
    }
#endif
}
