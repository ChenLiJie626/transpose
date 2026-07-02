# ComplexTranspose

Ascend C custom operator for transposing split complex matrices.

## Shape

Inputs:

- `a_r`: `float`, shape `[batch * 136, 8, 256]`
- `a_i`: `float`, shape `[batch * 136, 8, 256]`

Outputs:

- `c_r`: `float`, shape `[136, 256, 8 * batch]`
- `c_i`: `float`, shape `[136, 256, 8 * batch]`

Mapping:

```text
c_r[row, k, inner * batch + batch_idx] = a_r[batch_idx * 136 + row, inner, k]
c_i[row, k, inner * batch + batch_idx] = a_i[batch_idx * 136 + row, inner, k]
```

## Build

```bash
cd ComplexTransposeMultiCore
./build.sh
```

## AclNN Invocation

```bash
cd AclNNInvocation
./run.sh 10 0 10 100
```
