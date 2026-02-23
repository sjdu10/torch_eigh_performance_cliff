# torch_eigh_performance_cliff
Observation of performance cliff of pytorch batched eigh operation for different matrix size, and my solution to it.

## 1. Observation in default PyTorch `eigh` performance

`torch.linalg.eigh` applied to a batch of (B, n, n) symmetric matrices exhibits smooth scaling for n<=32, then an abrupt 80-140x slowdown at n=33.

**Table 1: Default `torch.linalg.eigh` timing (ms) vs. matrix size n.**

|  n  | B=64 time (ms) | B=64 ratio | B=1024 time (ms) | B=1024 ratio |
|----:|---------------:|-----------:|------------------:|-------------:|
|   8 |           0.29 |            |              0.91 |              |
|  16 |           0.52 |            |              2.86 |              |
|  24 |           0.85 |            |              5.49 |              |
|  28 |           0.93 |            |              7.22 |              |
|  32 |           1.05 |       1.0x |              8.30 |         1.0x |
|  33 |          77.20 |       73x  |              1137 |        137x  |
|  34 |          78.87 |       75x  |              1153 |        139x  |
|  40 |          87.37 |       83x  |              1296 |        156x  |
|  48 |          97.31 |       93x  |              1491 |        180x  |
|  64 |          122.1 |      116x  |              1874 |        226x  |
|  96 |          246.2 |      234x  |              3685 |        444x  |
| 128 |          316.0 |      301x  |              4783 |        576x  |

## 2. The fix: dispatch updated cuSOLVER batched `eigh` (`cusolverDnXsyevBatched`), as in JAX PR #31375

`cusolverDnXsyevBatched` (available since cuSOLVER 11.7.1 / CUDA 12.6) is a genuinely batched eigensolver (divide-and-conquer, or Jacobi for n<=32) with **no matrix-size limitation**. PyTorch already wraps it and uses it for n<=32. The fix is to remove the n<=32 gate:
```cpp
// Proposed change in linalg_eigh_cusolver:
  // cusolverDnXsyevBatched works for all n when batch size > 1.
  // See jax-ml/jax#31375 for precedent.
if (batchCount(eigenvectors) > 1) {
    linalg_eigh_cusolver_syevj_batched(...);
} else if (scalar_type == kFloat
            && size >= 32 && size <= 512) {
// Non-batched: preserve original dispatch.
    linalg_eigh_cusolver_syevj(...);
} else {
    linalg_eigh_cusolver_syevd(...);
}
```

This is similar to the approach JAX took in [jax-ml/jax#31375](https://github.com/jax-ml/jax/pull/31375) (merged September 2025): route all batched `eigh` through `cusolverDnXsyevBatched`, replacing the per-matrix `syevd` loop.

**Correctness.** In our test, the maximum eigenvalue difference |delta| w.r.t. results from default `torch.linalg.eigh` and eigenvector residual ||Av - lambda v|| are required to be smaller than 1e-10 to ensure the correctness of the eigen decomposition. For an example case at (B=16, n=48), we have verified |delta|=0 and ||Av - lambda v||=2.13e-13, which pass the correctness test.

**Performance.**
We compare the `eigh` performance for default `torch.linalg.eigh` and our proposed fix using `XsyevBatched` for batched matrix input, see Table 2&3.

**Table 2: Default `eigh` vs. `XsyevBatched` extension (B=64, `float64`).**

|  n  | default (ms) | XsyevBatched (ms) | speedup  |
|----:|-----------:|-------------------:|:---------|
|   8 |       0.29 |               0.20 | 1.5x     |
|  16 |       0.52 |               0.41 | 1.3x     |
|  24 |       0.85 |               0.74 | 1.1x     |
|  28 |       0.93 |               0.78 | 1.2x     |
|  32 |       1.05 |               0.88 | 1.2x     |
|  33 |      77.20 |               1.68 | **46x**  |
|  34 |      78.87 |               1.43 | **55x**  |
|  40 |      87.37 |               1.53 | **57x**  |
|  48 |      97.31 |               1.77 | **55x**  |
|  64 |      122.1 |               2.34 | **52x**  |
|  96 |      246.2 |               6.74 | **37x**  |
| 128 |      316.0 |               9.88 | **32x**  |

**Table 3: Default `eigh` vs. `XsyevBatched` extension (B=1024, `float64`).**

|  n  | default (ms) | XsyevBatched (ms) | speedup  |
|----:|-----------:|-------------------:|:---------|
|   8 |       0.91 |               0.75 | 1.2x     |
|  16 |       2.86 |               2.66 | 1.1x     |
|  24 |       5.49 |               5.47 | 1.0x     |
|  28 |       7.22 |               6.92 | 1.0x     |
|  32 |       8.30 |               7.81 | 1.1x     |
|  33 |       1137 |              15.06 | **75x**  |
|  34 |       1153 |              15.01 | **77x**  |
|  40 |       1296 |              17.36 | **75x**  |
|  48 |       1491 |              21.89 | **68x**  |
|  64 |       1874 |              30.44 | **62x**  |
|  96 |       3685 |              96.10 | **38x**  |
| 128 |       4783 |              145.7 | **33x**  |

## Summary

1. The n=32 to 33 performance cliff is **completely eliminated** --- timing scales smoothly across all n.
2. **32-77x speedup** for n>32, with no regression for n<=32.
3. **Numerically identical** to default PyTorch (same cuSOLVER API, just called unconditionally).
4. The fix is a **one-line dispatch change**, using an API PyTorch already wraps.
5. This matches what [JAX shipped in September 2025](https://github.com/jax-ml/jax/pull/31375).

## Appendix A: Reproducer

```python
import torch

device = "cuda"
B = 1024
dtype = torch.float64

for n in [31, 32, 33, 34, 48, 64]:
    x = torch.randn(B, n, n, device=device, dtype=dtype)
    a = (x + x.mT) / 2

    # warmup
    for _ in range(5):
        torch.linalg.eigh(a)
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(10):
        torch.linalg.eigh(a)
    end.record()
    torch.cuda.synchronize()
    t = start.elapsed_time(end) / 10
    print(f"eigh n={n}: {t:.2f} ms")
```

## Appendix B: C++ extension source (`my_eigh.cpp`)

```cpp
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <cusolverDn.h>

static cusolverDnHandle_t get_handle() {
    static thread_local cusolverDnHandle_t h = nullptr;
    if (!h) cusolverDnCreate(&h);
    cusolverDnSetStream(h,
        c10::cuda::getCurrentCUDAStream().stream());
    return h;
}

std::tuple<torch::Tensor, torch::Tensor>
custom_batched_eigh(torch::Tensor input, bool upper) {
    TORCH_CHECK(input.is_cuda(),
                "input must be a CUDA tensor");
    TORCH_CHECK(input.dim() >= 2
                && input.size(-1) == input.size(-2),
                "input must be square");

    const auto dtype = input.scalar_type();
    const int64_t n = input.size(-1);
    const int64_t batch = input.numel() / (n * n);

    // Clone: cuSOLVER overwrites input with eigenvectors.
    // Symmetric => row-major == column-major, no transpose needed.
    auto vectors = input.contiguous().clone();
    auto values = torch::empty({batch, n}, input.options());
    auto info = torch::zeros({batch},
        input.options().dtype(torch::kInt32));

    auto handle = get_handle();
    cusolverDnParams_t params;
    cusolverDnCreateParams(&params);

    auto jobz = CUSOLVER_EIG_MODE_VECTOR;
    auto uplo = upper ? CUBLAS_FILL_MODE_UPPER
                      : CUBLAS_FILL_MODE_LOWER;
    auto cuda_dtype = (dtype == torch::kFloat64)
                      ? CUDA_R_64F : CUDA_R_32F;

    // Query workspace, allocate, compute
    size_t work_device_sz = 0, work_host_sz = 0;
    cusolverDnXsyevBatched_bufferSize(
        handle, params, jobz, uplo, n, cuda_dtype,
        vectors.data_ptr(), n, cuda_dtype,
        values.data_ptr(), cuda_dtype,
        &work_device_sz, &work_host_sz, batch);

    auto work_device = torch::empty(
        {static_cast<int64_t>(work_device_sz)},
        input.options().dtype(torch::kUInt8));
    std::vector<uint8_t> work_host(work_host_sz);

    cusolverDnXsyevBatched(
        handle, params, jobz, uplo, n, cuda_dtype,
        vectors.data_ptr(), n, cuda_dtype,
        values.data_ptr(), cuda_dtype,
        work_device.data_ptr(), work_device_sz,
        work_host.data(), work_host_sz,
        info.data_ptr<int>(), batch);

    cusolverDnDestroyParams(params);

    // cuSOLVER writes column-major eigenvectors into our
    // row-major buffer; transpose back for PyTorch.
    vectors = vectors.view({batch, n, n})
                     .mT().contiguous();
    auto out_shape = input.sizes().vec();
    vectors = vectors.reshape(out_shape);
    auto val_shape = std::vector<int64_t>(
        out_shape.begin(), out_shape.end() - 1);
    values = values.reshape(val_shape);

    return {values, vectors};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("eigh", &custom_batched_eigh,
          "Batched eigh via cusolverDnXsyevBatched",
          py::arg("input"), py::arg("upper") = false);
}
```
