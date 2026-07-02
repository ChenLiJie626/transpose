#ifndef COMPLEX_TRANSPOSE_TILING_H
#define COMPLEX_TRANSPOSE_TILING_H

#include <cstdint>

#include "register/tilingdata_base.h"

namespace ComplexTransposeConst {
constexpr int32_t ROWS = 136;
constexpr int32_t INNER = 8;
constexpr int32_t COLS = 256;
constexpr int32_t DEFAULT_BLOCK_DIM = 16;
} // namespace ComplexTransposeConst

namespace optiling {
BEGIN_TILING_DATA_DEF(ComplexTransposeTilingData)
TILING_DATA_FIELD_DEF(uint32_t, batch);
TILING_DATA_FIELD_DEF(uint32_t, totalPlanes);
TILING_DATA_FIELD_DEF(uint32_t, lastDim);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(ComplexTranspose, ComplexTransposeTilingData)
} // namespace optiling

#endif // COMPLEX_TRANSPOSE_TILING_H
