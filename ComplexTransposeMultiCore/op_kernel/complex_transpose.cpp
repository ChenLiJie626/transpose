#include "kernel_operator.h"

using namespace AscendC;

namespace ComplexTransposeKernel {
constexpr uint32_t ROWS = 136;
constexpr uint32_t INNER = 8;
constexpr uint32_t COLS = 256;
constexpr uint32_t PLANE_ELEMS = INNER * COLS;
constexpr uint32_t BATCH_ROW_CHUNK = 16;
constexpr uint32_t INPUT_CHUNK_ELEMS = BATCH_ROW_CHUNK * INNER * COLS;
constexpr uint32_t OUTPUT_CHUNK_ELEMS = BATCH_ROW_CHUNK * INNER;
constexpr uint32_t INPUT_CHUNK_BYTES = INPUT_CHUNK_ELEMS * sizeof(float);
constexpr uint32_t OUTPUT_CHUNK_BYTES = OUTPUT_CHUNK_ELEMS * sizeof(float);

__aicore__ inline uint64_t GetAivBlockNum()
{
    return static_cast<uint64_t>(GetBlockNum()) * static_cast<uint64_t>(GetTaskRation());
}

__aicore__ inline void CopyInputBatchChunk(GlobalTensor<float> &src,
                                           LocalTensor<float> &dst,
                                           uint32_t row,
                                           uint32_t batchStart,
                                           uint32_t batchLen)
{
    for (uint32_t batchOffset = 0; batchOffset < batchLen; ++batchOffset) {
        const uint32_t batchIdx = batchStart + batchOffset;
        for (uint32_t inner = 0; inner < INNER; ++inner) {
            const uint64_t srcOffset = (static_cast<uint64_t>(batchIdx) * ROWS + row) * PLANE_ELEMS +
                                       static_cast<uint64_t>(inner) * COLS;
            const uint32_t dstOffset = (batchOffset * INNER + inner) * COLS;
            DataCopy(dst[dstOffset], src[srcOffset], COLS);
        }
    }
}

__aicore__ inline void FillFullLastDimBlock(LocalTensor<float> &src,
                                            LocalTensor<float> &dst,
                                            uint32_t batch,
                                            uint32_t col)
{
    for (uint32_t inner = 0; inner < INNER; ++inner) {
        for (uint32_t batchOffset = 0; batchOffset < batch; ++batchOffset) {
            const uint32_t srcOffset = (batchOffset * INNER + inner) * COLS + col;
            const uint32_t dstOffset = inner * batch + batchOffset;
            dst.SetValue(dstOffset, src.GetValue(srcOffset));
        }
    }
}

__aicore__ inline void FillInnerBatchChunk(LocalTensor<float> &src,
                                           LocalTensor<float> &dst,
                                           uint32_t inner,
                                           uint32_t batchLen,
                                           uint32_t col)
{
    for (uint32_t batchOffset = 0; batchOffset < batchLen; ++batchOffset) {
        const uint32_t srcOffset = (batchOffset * INNER + inner) * COLS + col;
        dst.SetValue(batchOffset, src.GetValue(srcOffset));
    }
}

__aicore__ inline void CopyLocalPad(GlobalTensor<float> &dst,
                                    uint64_t dstOffset,
                                    LocalTensor<float> &src,
                                    uint32_t len)
{
    DataCopyExtParams copyParams{1, static_cast<uint32_t>(len * sizeof(float)), 0, 0, 0};
    DataCopyPad(dst[dstOffset], src, copyParams);
}

__aicore__ inline void StoreFullBatchRow(GlobalTensor<float> &dst,
                                         LocalTensor<float> &inLocal,
                                         LocalTensor<float> &outLocal,
                                         uint32_t row,
                                         uint32_t batch,
                                         uint32_t lastDim)
{
    const uint64_t rowOutputBase = static_cast<uint64_t>(row) * COLS * lastDim;
    for (uint32_t col = 0; col < COLS; ++col) {
        FillFullLastDimBlock(inLocal, outLocal, batch, col);
        PipeBarrier<PIPE_ALL>();
        DataCopy(dst[rowOutputBase + static_cast<uint64_t>(col) * lastDim], outLocal, lastDim);
        PipeBarrier<PIPE_ALL>();
    }
}

__aicore__ inline void StoreBatchChunkRow(GlobalTensor<float> &dst,
                                          LocalTensor<float> &inLocal,
                                          LocalTensor<float> &outLocal,
                                          uint32_t row,
                                          uint32_t batchStart,
                                          uint32_t batchLen,
                                          uint32_t batch,
                                          uint32_t lastDim)
{
    const uint64_t rowOutputBase = static_cast<uint64_t>(row) * COLS * lastDim;
    for (uint32_t col = 0; col < COLS; ++col) {
        const uint64_t colOutputBase = rowOutputBase + static_cast<uint64_t>(col) * lastDim;
        for (uint32_t inner = 0; inner < INNER; ++inner) {
            FillInnerBatchChunk(inLocal, outLocal, inner, batchLen, col);
            PipeBarrier<PIPE_ALL>();
            CopyLocalPad(dst, colOutputBase + static_cast<uint64_t>(inner) * batch + batchStart,
                         outLocal, batchLen);
            PipeBarrier<PIPE_ALL>();
        }
    }
}

__aicore__ inline void ProcessInputPart(GlobalTensor<float> &src,
                                        GlobalTensor<float> &dst,
                                        LocalTensor<float> &inLocal,
                                        LocalTensor<float> &outLocal,
                                        uint32_t row,
                                        uint32_t batch,
                                        uint32_t lastDim)
{
    if (batch <= BATCH_ROW_CHUNK) {
        CopyInputBatchChunk(src, inLocal, row, 0, batch);
        PipeBarrier<PIPE_ALL>();
        StoreFullBatchRow(dst, inLocal, outLocal, row, batch, lastDim);
        return;
    }

    for (uint32_t batchStart = 0; batchStart < batch; batchStart += BATCH_ROW_CHUNK) {
        const uint32_t batchLen = (batchStart + BATCH_ROW_CHUNK <= batch) ?
                                  BATCH_ROW_CHUNK : (batch - batchStart);
        CopyInputBatchChunk(src, inLocal, row, batchStart, batchLen);
        PipeBarrier<PIPE_ALL>();
        StoreBatchChunkRow(dst, inLocal, outLocal, row, batchStart, batchLen, batch, lastDim);
    }
}

__aicore__ inline void ProcessRow(GlobalTensor<float> &arGm,
                                  GlobalTensor<float> &aiGm,
                                  GlobalTensor<float> &crGm,
                                  GlobalTensor<float> &ciGm,
                                  TBuf<> &inBuf,
                                  TBuf<> &outBuf,
                                  uint32_t row,
                                  uint32_t batch,
                                  uint32_t lastDim)
{
    LocalTensor<float> inLocal = inBuf.Get<float>();
    LocalTensor<float> outLocal = outBuf.Get<float>();
    ProcessInputPart(arGm, crGm, inLocal, outLocal, row, batch, lastDim);
    ProcessInputPart(aiGm, ciGm, inLocal, outLocal, row, batch, lastDim);
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
    TBuf<> inBuf;
    TBuf<> outBuf;
    pipe.InitBuffer(inBuf, INPUT_CHUNK_BYTES);
    pipe.InitBuffer(outBuf, OUTPUT_CHUNK_BYTES);

    const uint64_t blockIdx = static_cast<uint64_t>(GetBlockIdx());
    const uint64_t blockNum = GetAivBlockNum();
    for (uint64_t row = blockIdx; row < ROWS; row += blockNum) {
        ProcessRow(arGm, aiGm, crGm, ciGm, inBuf, outBuf,
                   static_cast<uint32_t>(row), batch, lastDim);
    }
#endif
}
