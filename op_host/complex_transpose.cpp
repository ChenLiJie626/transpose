#include "complex_transpose_tiling.h"

#include "exe_graph/runtime/infer_datatype_context.h"
#include "exe_graph/runtime/infer_shape_context.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <limits>

using namespace ComplexTransposeConst;

namespace {
#define CT_LOG_FAIL(stage, fmt, ...) \
    std::fprintf(stderr, "[ComplexTranspose][%s] " fmt "\n", stage, ##__VA_ARGS__)

void LogShape(const char *stage, const char *name, const gert::Shape &shape)
{
    std::fprintf(stderr, "[ComplexTranspose][%s] %s rank=%zu dims=[",
                 stage, name, shape.GetDimNum());
    for (size_t i = 0; i < shape.GetDimNum(); ++i) {
        std::fprintf(stderr, "%s%ld", i == 0 ? "" : ",", static_cast<long>(shape.GetDim(i)));
    }
    std::fprintf(stderr, "]\n");
}

bool IsDynamicDim(int64_t dim)
{
    return dim < 0;
}

bool CheckSameOrDynamicDim(int64_t lhs, int64_t rhs)
{
    return lhs == rhs || IsDynamicDim(lhs) || IsDynamicDim(rhs);
}

bool CheckInputShapes(const gert::Shape &arShape, const gert::Shape &aiShape,
                      bool requireStaticBatch, int64_t &batch, const char *stage)
{
    batch = -1;
    if (arShape.GetDimNum() != 3) {
        CT_LOG_FAIL(stage, "invalid a_r rank, expect rank=3");
        LogShape(stage, "a_r", arShape);
        return false;
    }
    if (arShape.GetDim(1) != INNER || arShape.GetDim(2) != COLS) {
        CT_LOG_FAIL(stage, "invalid a_r shape tail, expect [*,%d,%d]", INNER, COLS);
        LogShape(stage, "a_r", arShape);
        return false;
    }

    const int64_t firstDim = arShape.GetDim(0);
    if (IsDynamicDim(firstDim)) {
        if (requireStaticBatch) {
            CT_LOG_FAIL(stage, "a_r dim0 must be concrete at tiling time");
            LogShape(stage, "a_r", arShape);
            return false;
        }
    } else if (firstDim <= 0 || firstDim % ROWS != 0) {
        CT_LOG_FAIL(stage, "invalid a_r dim0, expect positive batch * %d, actual %ld",
                    ROWS, static_cast<long>(firstDim));
        LogShape(stage, "a_r", arShape);
        return false;
    } else {
        batch = firstDim / ROWS;
    }

    if (aiShape.GetDimNum() != 3 ||
        !CheckSameOrDynamicDim(aiShape.GetDim(0), arShape.GetDim(0)) ||
        aiShape.GetDim(1) != INNER ||
        aiShape.GetDim(2) != COLS) {
        CT_LOG_FAIL(stage, "invalid a_i shape, expect same dim0 as a_r and tail [%d,%d]",
                    INNER, COLS);
        LogShape(stage, "a_i", aiShape);
        return false;
    }
    if (requireStaticBatch && aiShape.GetDim(0) != arShape.GetDim(0)) {
        CT_LOG_FAIL(stage, "a_i dim0 must equal a_r dim0 at tiling time");
        LogShape(stage, "a_i", aiShape);
        return false;
    }
    return true;
}

bool CheckOutputShape(const gert::Shape &shape, int64_t batch)
{
    return shape.GetDimNum() == 3 &&
           CheckSameOrDynamicDim(shape.GetDim(0), ROWS) &&
           CheckSameOrDynamicDim(shape.GetDim(1), COLS) &&
           CheckSameOrDynamicDim(shape.GetDim(2), batch * INNER);
}

static ge::graphStatus InferShape(gert::InferShapeContext *context)
{
    constexpr const char *stage = "InferShape";
    if (context == nullptr) {
        CT_LOG_FAIL(stage, "context is nullptr");
        return ge::GRAPH_FAILED;
    }

    const gert::Shape *arShape = context->GetInputShape(0);
    const gert::Shape *aiShape = context->GetInputShape(1);
    gert::Shape *crShape = context->GetOutputShape(0);
    gert::Shape *ciShape = context->GetOutputShape(1);
    if (arShape == nullptr || aiShape == nullptr || crShape == nullptr || ciShape == nullptr) {
        CT_LOG_FAIL(stage, "null shape pointer: a_r=%p a_i=%p c_r=%p c_i=%p",
                    static_cast<const void *>(arShape), static_cast<const void *>(aiShape),
                    static_cast<void *>(crShape), static_cast<void *>(ciShape));
        return ge::GRAPH_FAILED;
    }

    int64_t batch = -1;
    if (!CheckInputShapes(*arShape, *aiShape, false, batch, stage)) {
        return ge::GRAPH_FAILED;
    }

    const int64_t lastDim = batch < 0 ? -1 : batch * INNER;
    *crShape = gert::Shape({ROWS, COLS, lastDim});
    *ciShape = gert::Shape({ROWS, COLS, lastDim});
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferDataType(gert::InferDataTypeContext *context)
{
    constexpr const char *stage = "InferDataType";
    if (context == nullptr) {
        CT_LOG_FAIL(stage, "context is nullptr");
        return ge::GRAPH_FAILED;
    }
    if (context->GetInputDataType(0) != ge::DT_FLOAT ||
        context->GetInputDataType(1) != ge::DT_FLOAT) {
        CT_LOG_FAIL(stage, "a_r and a_i must be DT_FLOAT");
        return ge::GRAPH_FAILED;
    }
    if (context->SetOutputDataType(0, ge::DT_FLOAT) != ge::GRAPH_SUCCESS ||
        context->SetOutputDataType(1, ge::DT_FLOAT) != ge::GRAPH_SUCCESS) {
        CT_LOG_FAIL(stage, "SetOutputDataType failed");
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}
} // namespace

namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext *context)
{
    constexpr const char *stage = "TilingFunc";
    if (context == nullptr) {
        CT_LOG_FAIL(stage, "context is nullptr");
        return ge::GRAPH_FAILED;
    }

    auto arTensor = context->GetInputTensor(0);
    auto aiTensor = context->GetInputTensor(1);
    auto crStorageShape = context->GetOutputShape(0);
    auto ciStorageShape = context->GetOutputShape(1);
    if (arTensor == nullptr || aiTensor == nullptr ||
        crStorageShape == nullptr || ciStorageShape == nullptr) {
        CT_LOG_FAIL(stage, "null tensor/shape pointer");
        return ge::GRAPH_FAILED;
    }

    auto arShape = arTensor->GetOriginShape();
    auto aiShape = aiTensor->GetOriginShape();
    auto crShape = crStorageShape->GetOriginShape();
    auto ciShape = ciStorageShape->GetOriginShape();

    int64_t batch = -1;
    if (!CheckInputShapes(arShape, aiShape, true, batch, stage)) {
        return ge::GRAPH_FAILED;
    }
    if (!CheckOutputShape(crShape, batch) || !CheckOutputShape(ciShape, batch)) {
        CT_LOG_FAIL(stage, "invalid output shape, expect [%d,%d,%ld]",
                    ROWS, COLS, static_cast<long>(batch * INNER));
        LogShape(stage, "c_r", crShape);
        LogShape(stage, "c_i", ciShape);
        return ge::GRAPH_FAILED;
    }
    if (batch <= 0 ||
        batch > static_cast<int64_t>(std::numeric_limits<uint32_t>::max() / INNER) ||
        batch > static_cast<int64_t>(std::numeric_limits<uint32_t>::max() / ROWS)) {
        CT_LOG_FAIL(stage, "batch out of uint32 range, batch=%ld", static_cast<long>(batch));
        return ge::GRAPH_FAILED;
    }

    ComplexTransposeTilingData tiling;
    tiling.set_batch(static_cast<uint32_t>(batch));
    tiling.set_totalPlanes(static_cast<uint32_t>(batch * ROWS));
    tiling.set_lastDim(static_cast<uint32_t>(batch * INNER));

    const uint32_t totalPlanes = static_cast<uint32_t>(batch * ROWS);
    const uint32_t blockDim = totalPlanes < DEFAULT_BLOCK_DIM ? totalPlanes : DEFAULT_BLOCK_DIM;
    context->SetBlockDim(blockDim);
    context->SetTilingKey(0);

    auto rawTilingData = context->GetRawTilingData();
    if (rawTilingData == nullptr) {
        CT_LOG_FAIL(stage, "raw tiling data is nullptr");
        return ge::GRAPH_FAILED;
    }
    tiling.SaveToBuffer(rawTilingData->GetData(), rawTilingData->GetCapacity());
    rawTilingData->SetDataSize(tiling.GetDataSize());

    size_t *currentWorkspace = context->GetWorkspaceSizes(1);
    if (currentWorkspace == nullptr) {
        CT_LOG_FAIL(stage, "workspace size pointer is nullptr");
        return ge::GRAPH_FAILED;
    }
    currentWorkspace[0] = 0;
    return ge::GRAPH_SUCCESS;
}
} // namespace optiling

namespace ops {
class ComplexTranspose : public OpDef {
public:
    explicit ComplexTranspose(const char *name) : OpDef(name)
    {
        this->SetInferShape(InferShape)
            .SetInferDataType(InferDataType);

        this->Input("a_r")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND});
        this->Input("a_i")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND});
        this->Output("c_r")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND});
        this->Output("c_i")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND});

        this->AICore()
            .SetTiling(optiling::TilingFunc)
            .AddConfig("ascend910b")
            .AddConfig("ascend910_93");
    }
};

OP_ADD(ComplexTranspose);
} // namespace ops
