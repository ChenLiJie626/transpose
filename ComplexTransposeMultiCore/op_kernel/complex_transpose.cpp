#include "kernel_operator.h"
#include "adv_api/index/arithprogression.h"

using namespace AscendC;

namespace ComplexTransposeKernel {
constexpr uint32_t ROWS = 136;
constexpr uint32_t INNER = 8;
constexpr uint32_t COLS = 256;
constexpr uint32_t PLANE_ELEMS = INNER * COLS;
constexpr uint32_t BUFFER_NUM = 2;
constexpr uint32_t BATCH_LANES = 8;
constexpr uint32_t CHUNK_ELEMS = BATCH_LANES * COLS;
constexpr uint32_t CHUNK_BYTES = CHUNK_ELEMS * sizeof(float);
constexpr uint32_t OFFSET_BLOCK_ELEMS = 32;
constexpr uint32_t OFFSET_BLOCK_COLS = OFFSET_BLOCK_ELEMS / BATCH_LANES;
constexpr uint32_t OFFSET_WORK_TENSORS = 2;
constexpr uint32_t OFFSET_WORK_BYTES = OFFSET_WORK_TENSORS * CHUNK_ELEMS * sizeof(uint32_t);

__aicore__ inline uint64_t GetAivBlockNum()
{
    return static_cast<uint64_t>(GetBlockNum()) * static_cast<uint64_t>(GetTaskRation());
}

__aicore__ inline void InitGatherOffsets(TBuf<> &offsetBuf)
{
    LocalTensor<int32_t> work = offsetBuf.Get<int32_t>();
    LocalTensor<int32_t> offset = work;
    LocalTensor<int32_t> lane = work[CHUNK_ELEMS];

    Arange(lane, static_cast<int32_t>(0), static_cast<int32_t>(1), static_cast<int32_t>(BATCH_LANES));
    Muls(lane, lane, static_cast<int32_t>(COLS), static_cast<int32_t>(BATCH_LANES));
    PipeBarrier<PIPE_V>();

    for (uint32_t colBase = 0; colBase < COLS; colBase += OFFSET_BLOCK_COLS) {
        const uint32_t base = colBase * BATCH_LANES;
        for (uint32_t col = 0; col < OFFSET_BLOCK_COLS; ++col) {
            LocalTensor<int32_t> offsetPart = offset[base + col * BATCH_LANES];
            Adds(offsetPart, lane, static_cast<int32_t>(colBase + col), static_cast<int32_t>(BATCH_LANES));
            Muls(offsetPart, offsetPart, static_cast<int32_t>(sizeof(float)), static_cast<int32_t>(BATCH_LANES));
        }
    }
    PipeBarrier<PIPE_V>();
}

__aicore__ inline void DecodeTask(uint32_t task,
                                  uint32_t batchChunks,
                                  uint32_t &inner,
                                  uint32_t &batchStart)
{
    inner = task / batchChunks;
    batchStart = (task - inner * batchChunks) * BATCH_LANES;
}

__aicore__ inline uint64_t GetInputOffset(uint32_t row,
                                          uint32_t inner,
                                          uint32_t batchIdx)
{
    return (static_cast<uint64_t>(batchIdx) * ROWS + row) * PLANE_ELEMS +
           static_cast<uint64_t>(inner) * COLS;
}

__aicore__ inline uint64_t GetOutputOffset(uint32_t row,
                                           uint32_t inner,
                                           uint32_t batchIdx,
                                           uint32_t batch,
                                           uint32_t lastDim)
{
    return static_cast<uint64_t>(row) * COLS * lastDim +
           static_cast<uint64_t>(inner) * batch + batchIdx;
}

__aicore__ inline void CopyInOne(GlobalTensor<float> &src,
                                 TQue<QuePosition::VECIN, BUFFER_NUM> &inQueue,
                                 uint32_t row,
                                 uint32_t inner,
                                 uint32_t batchStart,
                                 uint32_t batchLen)
{
    LocalTensor<float> inLocal = inQueue.AllocTensor<float>();
    for (uint32_t lane = 0; lane < batchLen; ++lane) {
        DataCopy(inLocal[lane * COLS], src[GetInputOffset(row, inner, batchStart + lane)], COLS);
    }
    inQueue.EnQue(inLocal);
}

__aicore__ inline void CopyInPair(GlobalTensor<float> &arGm,
                                  GlobalTensor<float> &aiGm,
                                  TQue<QuePosition::VECIN, BUFFER_NUM> &rInQueue,
                                  TQue<QuePosition::VECIN, BUFFER_NUM> &iInQueue,
                                  uint32_t row,
                                  uint32_t task,
                                  uint32_t batch,
                                  uint32_t batchChunks)
{
    uint32_t inner = 0;
    uint32_t batchStart = 0;
    DecodeTask(task, batchChunks, inner, batchStart);
    const uint32_t batchLen = (batchStart + BATCH_LANES <= batch) ? BATCH_LANES : (batch - batchStart);
    CopyInOne(arGm, rInQueue, row, inner, batchStart, batchLen);
    CopyInOne(aiGm, iInQueue, row, inner, batchStart, batchLen);
}

__aicore__ inline void MoveOne(TQue<QuePosition::VECIN, BUFFER_NUM> &inQueue,
                               TQue<QuePosition::VECOUT, BUFFER_NUM> &outQueue,
                               LocalTensor<uint32_t> &offsetLocal)
{
    LocalTensor<float> inLocal = inQueue.DeQue<float>();
    LocalTensor<float> outLocal = outQueue.AllocTensor<float>();
    for (uint32_t base = 0; base < CHUNK_ELEMS; base += OFFSET_BLOCK_ELEMS) {
        LocalTensor<float> outPart = outLocal[base];
        LocalTensor<uint32_t> offsetPart = offsetLocal[base];
        Gather(outPart, inLocal, offsetPart, 0, static_cast<uint64_t>(OFFSET_BLOCK_ELEMS),
               static_cast<uint8_t>(1), static_cast<uint16_t>(8));
    }
    outQueue.EnQue(outLocal);
    inQueue.FreeTensor(inLocal);
}

__aicore__ inline void MovePair(TQue<QuePosition::VECIN, BUFFER_NUM> &rInQueue,
                                TQue<QuePosition::VECIN, BUFFER_NUM> &iInQueue,
                                TQue<QuePosition::VECOUT, BUFFER_NUM> &rOutQueue,
                                TQue<QuePosition::VECOUT, BUFFER_NUM> &iOutQueue,
                                LocalTensor<uint32_t> &offsetLocal)
{
    MoveOne(rInQueue, rOutQueue, offsetLocal);
    MoveOne(iInQueue, iOutQueue, offsetLocal);
}

__aicore__ inline void CopyOutOne(GlobalTensor<float> &dst,
                                  TQue<QuePosition::VECOUT, BUFFER_NUM> &outQueue,
                                  uint32_t row,
                                  uint32_t inner,
                                  uint32_t batchStart,
                                  uint32_t batchLen,
                                  uint32_t batch,
                                  uint32_t lastDim)
{
    LocalTensor<float> outLocal = outQueue.DeQue<float>();
    DataCopyExtParams copyParams{static_cast<uint16_t>(COLS),
                                 static_cast<uint32_t>(batchLen * sizeof(float)),
                                 0,
                                 static_cast<uint32_t>((lastDim - batchLen) * sizeof(float)),
                                 0};
    DataCopyPad(dst[GetOutputOffset(row, inner, batchStart, batch, lastDim)], outLocal, copyParams);
    outQueue.FreeTensor(outLocal);
}

__aicore__ inline void CopyOutPair(GlobalTensor<float> &crGm,
                                   GlobalTensor<float> &ciGm,
                                   TQue<QuePosition::VECOUT, BUFFER_NUM> &rOutQueue,
                                   TQue<QuePosition::VECOUT, BUFFER_NUM> &iOutQueue,
                                   uint32_t row,
                                   uint32_t task,
                                   uint32_t batch,
                                   uint32_t batchChunks,
                                   uint32_t lastDim)
{
    uint32_t inner = 0;
    uint32_t batchStart = 0;
    DecodeTask(task, batchChunks, inner, batchStart);
    const uint32_t batchLen = (batchStart + BATCH_LANES <= batch) ? BATCH_LANES : (batch - batchStart);
    CopyOutOne(crGm, rOutQueue, row, inner, batchStart, batchLen, batch, lastDim);
    CopyOutOne(ciGm, iOutQueue, row, inner, batchStart, batchLen, batch, lastDim);
}

__aicore__ inline void ProcessRow(GlobalTensor<float> &arGm,
                                  GlobalTensor<float> &aiGm,
                                  GlobalTensor<float> &crGm,
                                  GlobalTensor<float> &ciGm,
                                  TQue<QuePosition::VECIN, BUFFER_NUM> &rInQueue,
                                  TQue<QuePosition::VECIN, BUFFER_NUM> &iInQueue,
                                  TQue<QuePosition::VECOUT, BUFFER_NUM> &rOutQueue,
                                  TQue<QuePosition::VECOUT, BUFFER_NUM> &iOutQueue,
                                  TBuf<> &offsetBuf,
                                  uint32_t row,
                                  uint32_t batch,
                                  uint32_t lastDim)
{
    const uint32_t batchChunks = (batch + BATCH_LANES - 1) / BATCH_LANES;
    const uint32_t totalTasks = INNER * batchChunks;
    LocalTensor<uint32_t> offsetLocal = offsetBuf.Get<uint32_t>();
    const uint32_t warmupTasks = (totalTasks < BUFFER_NUM) ? totalTasks : BUFFER_NUM;
    for (uint32_t task = 0; task < warmupTasks; ++task) {
        CopyInPair(arGm, aiGm, rInQueue, iInQueue, row, task, batch, batchChunks);
    }

    for (uint32_t task = 0; task < totalTasks; ++task) {
        MovePair(rInQueue, iInQueue, rOutQueue, iOutQueue, offsetLocal);

        const uint32_t nextTask = task + BUFFER_NUM;
        if (nextTask < totalTasks) {
            CopyInPair(arGm, aiGm, rInQueue, iInQueue, row, nextTask, batch, batchChunks);
        }

        CopyOutPair(crGm, ciGm, rOutQueue, iOutQueue, row, task, batch, batchChunks, lastDim);
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
    TQue<QuePosition::VECIN, BUFFER_NUM> rInQueue;
    TQue<QuePosition::VECIN, BUFFER_NUM> iInQueue;
    TQue<QuePosition::VECOUT, BUFFER_NUM> rOutQueue;
    TQue<QuePosition::VECOUT, BUFFER_NUM> iOutQueue;
    TBuf<> offsetBuf;
    pipe.InitBuffer(rInQueue, BUFFER_NUM, CHUNK_BYTES);
    pipe.InitBuffer(iInQueue, BUFFER_NUM, CHUNK_BYTES);
    pipe.InitBuffer(rOutQueue, BUFFER_NUM, CHUNK_BYTES);
    pipe.InitBuffer(iOutQueue, BUFFER_NUM, CHUNK_BYTES);
    pipe.InitBuffer(offsetBuf, OFFSET_WORK_BYTES);
    InitGatherOffsets(offsetBuf);

    const uint64_t blockIdx = static_cast<uint64_t>(GetBlockIdx());
    const uint64_t blockNum = GetAivBlockNum();
    for (uint64_t row = blockIdx; row < ROWS; row += blockNum) {
        ProcessRow(arGm, aiGm, crGm, ciGm, rInQueue, iInQueue, rOutQueue, iOutQueue, offsetBuf,
                   static_cast<uint32_t>(row), batch, lastDim);
    }
#endif
}
