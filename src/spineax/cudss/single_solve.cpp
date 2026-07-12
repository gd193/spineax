/*Standard single solve*/

#include <cstdlib>
#include <cstdint>
#include <memory>
#include <mutex>
#include <vector>
#include <complex>
#include <type_traits>
#include <string>
// #include <cuComplex.h> // For device-side complex number operations - not needed now that I am not writing kernels

#include "cuda_runtime_api.h"
#include "nanobind/nanobind.h"
#include "xla/ffi/api/ffi.h"
#include "cudss.h"
#include "cudss_env_options.h"

namespace ffi = xla::ffi;
namespace nb = nanobind;

// verification ================================================================
#define CUDSS_CALL_AND_CHECK(call, status, msg) \
    do { \
        status = call; \
        if (status != CUDSS_STATUS_SUCCESS) { \
            printf("FAILED: CUDSS call ended unsuccessfully with status = %d, details: " #msg "\n", status); \
            return ffi::Error::Internal(std::string("cuDSS call failed with status ") + \
                std::to_string(status) + ": " #msg); \
        } \
    } while(0);

#define CUDA_CHECK(call)                                       \
  do {                                                         \
    cudaError_t err = call;                                    \
    if (err != cudaSuccess) {                                  \
      printf("CUDA Error at %s %d: %s\n", __FILE__, __LINE__,   \
             cudaGetErrorString(err));                         \
      return ffi::Error::Internal(std::string("CUDA call failed: ") + cudaGetErrorString(err)); \
    }                                                          \
  } while (0)

#define CUDA_LOG_IF_ERROR(call)                                \
  do {                                                         \
    cudaError_t err = call;                                    \
    if (err != cudaSuccess) {                                  \
      printf("CUDA Error at %s %d: %s\n", __FILE__, __LINE__,   \
             cudaGetErrorString(err));                         \
    }                                                          \
  } while (0)


// Debugging functions =========================================================
template <typename T>
void print_device_data(
    const char* label,
    void* device_ptr,
    size_t n_batch,
    size_t n_elements_per_batch)
{
    // Ensure we have a valid pointer and something to print
    if (!device_ptr || n_batch == 0 || n_elements_per_batch == 0) return;

    std::cout << "\n--- Debug Print: " << label << " ---" << std::endl;

    // Calculate total size and create a host-side vector
    size_t total_elements = n_batch * n_elements_per_batch;
    std::vector<T> host_data(total_elements);

    // Copy all data from GPU to CPU in one go
    CUDA_LOG_IF_ERROR(cudaMemcpy(
        host_data.data(),
        device_ptr,
        total_elements * sizeof(T),
        cudaMemcpyDeviceToHost
    ));

    // Loop through each batch and print its contents
    for (size_t i = 0; i < n_batch; ++i) {
        std::cout << "Batch " << i << ": [";
        size_t batch_start_index = i * n_elements_per_batch;
        for (size_t j = 0; j < n_elements_per_batch; ++j) {
            std::cout << host_data[batch_start_index + j];
            if (j < n_elements_per_batch - 1) {
                std::cout << ", ";
            }
        }
        std::cout << "]" << std::endl;
    }
    std::cout << "------------------------------------" << std::endl;
}

// Helper function for data types ==============================================
template <ffi::DataType T> cudaDataType get_cuda_data_type();
template<> cudaDataType get_cuda_data_type<ffi::F32>() { return CUDA_R_32F; }
template<> cudaDataType get_cuda_data_type<ffi::F64>() { return CUDA_R_64F; }
template<> cudaDataType get_cuda_data_type<ffi::C64>() { return CUDA_C_32F; }
template<> cudaDataType get_cuda_data_type<ffi::C128>() { return CUDA_C_64F; }

template <ffi::DataType T>
struct get_native_data_type;
template<> struct get_native_data_type<ffi::F32> { using type = float; };
template<> struct get_native_data_type<ffi::F64> { using type = double; };
template<> struct get_native_data_type<ffi::C64> { using type = std::complex<float>; };
template<> struct get_native_data_type<ffi::C128> { using type = std::complex<double>; };

// Structure definitions =======================================================
template <ffi::DataType T>
struct CudssState {
    static xla::ffi::TypeId id;
    cudssHandle_t handle = nullptr;
    cudssConfig_t config = nullptr;
    cudssData_t data = nullptr;
    cudssMatrix_t A = nullptr;
    cudssMatrix_t x = nullptr;
    cudssMatrix_t b = nullptr;
    cudssMatrixType_t mtype = CUDSS_MTYPE_SYMMETRIC;
    cudssMatrixViewType_t mview = CUDSS_MVIEW_UPPER;
    cudssIndexBase_t base = CUDSS_BASE_ZERO;
    cudssStatus_t status = CUDSS_STATUS_SUCCESS;
    cudaStream_t last_stream = nullptr;
    int64_t device_id = -1;
    int64_t n = 0;
    int64_t nnz = 0;
    int64_t nrhs = 0;
    int64_t call_count = 0;
    size_t sizeWritten = 0;
    cudaDataType cuda_dtype = get_cuda_data_type<T>();
    using native_dtype = typename get_native_data_type<T>::type;
    int32_t* owned_csr_offsets = nullptr;
    int32_t* owned_csr_columns = nullptr;
    native_dtype* owned_csr_values = nullptr;
    size_t owned_csr_offsets_bytes = 0;
    size_t owned_csr_columns_bytes = 0;
    size_t owned_csr_values_bytes = 0;

    // cuDSS handle/data/descriptors are mutable and not documented as safe for
    // concurrent solves. Only constant handlers take this lock; dynamic hot
    // paths retain their existing lock-free behavior.
    std::mutex constant_mutex;

    void DestroyResources() noexcept {
        if (device_id >= 0) CUDA_LOG_IF_ERROR(cudaSetDevice(static_cast<int>(device_id)));
        if (last_stream) CUDA_LOG_IF_ERROR(cudaStreamSynchronize(last_stream));

        // Descriptors retain matrix pointers, so destroy them before buffers.
        if (A) { cudssMatrixDestroy(A); A = nullptr; }
        if (b) { cudssMatrixDestroy(b); b = nullptr; }
        if (x) { cudssMatrixDestroy(x); x = nullptr; }
        if (handle && data) { cudssDataDestroy(handle, data); data = nullptr; }
        if (config) { cudssConfigDestroy(config); config = nullptr; }
        if (handle) { cudssDestroy(handle); handle = nullptr; }
        if (owned_csr_offsets) { CUDA_LOG_IF_ERROR(cudaFree(owned_csr_offsets)); owned_csr_offsets = nullptr; }
        if (owned_csr_columns) { CUDA_LOG_IF_ERROR(cudaFree(owned_csr_columns)); owned_csr_columns = nullptr; }
        if (owned_csr_values) { CUDA_LOG_IF_ERROR(cudaFree(owned_csr_values)); owned_csr_values = nullptr; }
        owned_csr_offsets_bytes = owned_csr_columns_bytes = owned_csr_values_bytes = 0;
        n = nnz = 0;
        nrhs = 1;
        call_count = 0;
        last_stream = nullptr;
    }

    ~CudssState() {
        std::lock_guard<std::mutex> lock(constant_mutex);
        DestroyResources();
    }
};

template <> ffi::TypeId CudssState<ffi::F32>::id = {};
template <> ffi::TypeId CudssState<ffi::F64>::id = {};
template <> ffi::TypeId CudssState<ffi::C64>::id = {};
template <> ffi::TypeId CudssState<ffi::C128>::id = {};

// instantiation ===============================================================

// instantiate everything that is not a function of the context (cudaStream_t)
template <ffi::DataType T>
static ffi::ErrorOr<std::unique_ptr<CudssState<T>>> CudssInstantiate(
    // int32_t* offsets_ptr,                   // pointers and sizes of csr structure defn
    // const int64_t offsets_size,             // pointers and sizes of csr structure defn
    // int32_t* columns_ptr,                   // pointers and sizes of csr structure defn
    // const int64_t columns_size,             // pointers and sizes of csr structure defn
    const int64_t device_id,                // the device to run this on
    const int64_t mtype_id,                 // {0: gen, 1: sym, 2: herm, 3: spd, 4: hpd}
    const int64_t mview_id                  // {0: full, 1: triu, 2: tril}
) {

    // make a new state which will manage CuDSS's state
    auto state = std::make_unique<CudssState<T>>();

    // check on the type of matrix being solved
    if (mtype_id == 0) {
        // printf("general matrix chosen\n");
        state->mtype = CUDSS_MTYPE_GENERAL;
    } else if (mtype_id == 1) {
        // printf("symmetric matrix chosen\n");
        state->mtype = CUDSS_MTYPE_SYMMETRIC;
    } else if (mtype_id == 2) {
        // printf("hermitian matrix chosen\n");
        state->mtype = CUDSS_MTYPE_HERMITIAN;
    } else if (mtype_id == 3) {
        // printf("symmetric PD matrix chosen\n");
        state->mtype = CUDSS_MTYPE_SPD;
    } else if (mtype_id == 4) {
        // printf("hermitian PD matrix chosen\n");
        state->mtype = CUDSS_MTYPE_HPD;
    } else {
        throw std::invalid_argument("Invalid mtype_id. Valid options: 0: general, 1: symmetric, 2: hermitian, 3: spd, 4: hpd");
    }

    // check on the view of the matrix provided
    if (mview_id == 0) {
        // printf("full view provided\n");
        state->mview = CUDSS_MVIEW_FULL;
    } else if (mview_id == 1) {
        // printf("upper view provided\n");
        state->mview = CUDSS_MVIEW_UPPER;
    } else if (mview_id == 2) {
        // printf("lower view provided\n");
        state->mview = CUDSS_MVIEW_LOWER;
    } else {
        throw std::invalid_argument("Invalid mview_id. Valid options: 0: full, 1: upper, 2: lower");
    }

    // may as well store these for later for readability
    state->device_id = device_id;
    state->nrhs = 1; // the non-batched case

    // CUDA setup. Instantiation is transactional through unique_ptr.
    cudaError_t cuda_status = cudaSetDevice(static_cast<int>(device_id));
    if (cuda_status != cudaSuccess) {
        return ffi::Unexpected(
            ffi::Error::Internal(std::string("cudaSetDevice failed: ") +
                                 cudaGetErrorString(cuda_status)));
    }

    return state;
}

// execution ===================================================================
template <ffi::DataType T>
static ffi::Error CudssExecute(
    cudaStream_t stream,                    // JAXs stream given to this context (jit)
    CudssState<T>* state,                   // the state we instantiated in CudssInstantiate
    ffi::Buffer<T> b_values_buf,            // the real input data that varies per solution
    ffi::Buffer<T> csr_values_buf,          // the real input data that varies per solution
    ffi::Buffer<ffi::S32> offsets_buf,
    ffi::Buffer<ffi::S32> columns_buf,
    ffi::ResultBuffer<T> out_values_buf,    // the output buffer we write the answer to
    ffi::ResultBuffer<ffi::S32> inertia_buf,// the output buffer we write inertia diagnostics to
    const int64_t device_id,                // the device to run this on
    const int64_t mtype_id,                 // {0: gen, 1: sym, 2: herm, 3: spd, 4: hpd}
    const int64_t mview_id                  // {0: full, 1: triu, 2: tril}
) {

    // Track stream for cleanup synchronization
    state->last_stream = stream;

    // instantiate system branch
    if (state->call_count == 0) {

        // figure this out on first call
        state->n = offsets_buf.element_count() - 1;
        state->nnz = columns_buf.element_count();

        // CuDSS setup
        CUDSS_CALL_AND_CHECK(cudssCreate(&state->handle), state->status, "cudssCreate");
        CUDSS_CALL_AND_CHECK(cudssSetStream(state->handle, stream), state->status, "cudssSetStream");
        CUDSS_CALL_AND_CHECK(cudssConfigCreate(&state->config), state->status, "cudssConfigCreate");
        CUDSS_CALL_AND_CHECK(cudssDataCreate(state->handle, &state->data), state->status, "cudssDataCreate");

        // CuDSS structures creation
        CUDSS_CALL_AND_CHECK(cudssMatrixCreateDn(&state->b, state->n, state->nrhs, state->n,
            b_values_buf.typed_data(), state->cuda_dtype, CUDSS_LAYOUT_COL_MAJOR), state->status, "cudssMatrixCreateDn for b");

        CUDSS_CALL_AND_CHECK(cudssMatrixCreateDn(&state->x, state->n, state->nrhs, state->n,
            out_values_buf->typed_data(), state->cuda_dtype, CUDSS_LAYOUT_COL_MAJOR), state->status, "cudssMatrixCreateDn for x");

        CUDSS_CALL_AND_CHECK(cudssMatrixCreateCsr(&state->A, state->n, state->n, state->nnz,
            offsets_buf.typed_data(), NULL,
            columns_buf.typed_data(),
            csr_values_buf.typed_data(),
            CUDA_R_32I, state->cuda_dtype,
            state->mtype, state->mview, state->base), state->status, "cudssMatrixCreateCsr");

        // CuDSS config
        // iterative refinement of the soln is pretty n i f t y
        int iter_ref_nsteps = cudss_ir_nsteps();
        CUDSS_CALL_AND_CHECK(cudssConfigSet(state->config, CUDSS_CONFIG_IR_N_STEPS,
                            &iter_ref_nsteps, sizeof(iter_ref_nsteps)), state->status, "cudssConfigSet ir_nsteps");
        {
            ffi::Error env_error = cudss_apply_env_options_or_error(state->config);
            if (env_error.failure()) return env_error;
        }

        // cold solve - analyze, factorize, solve
        CUDSS_CALL_AND_CHECK(cudssExecute(state->handle, CUDSS_PHASE_ANALYSIS,
            state->config, state->data, state->A, state->x, state->b), state->status, "cudssExecute analysis");

        CUDSS_CALL_AND_CHECK(cudssExecute(state->handle, CUDSS_PHASE_FACTORIZATION,
            state->config, state->data, state->A, state->x, state->b), state->status, "cudssExecute factorization");

        CUDSS_CALL_AND_CHECK(cudssExecute(state->handle, CUDSS_PHASE_SOLVE,
            state->config, state->data, state->A, state->x, state->b), state->status, "cudssExecute solve");

        state->call_count++;
    }
    else {
        // stream can change between calls!!!
        CUDSS_CALL_AND_CHECK(cudssSetStream(state->handle, stream), state->status, "cudssSetStream");

        // set the values of the matrices - different to my batched solution
        // CUDSS_CALL_AND_CHECK(cudssMatrixSetValues(state->A, csr_values_buf.typed_data()), state->status, "update_pointers A");
        CUDSS_CALL_AND_CHECK(cudssMatrixSetCsrPointers(state->A,
            offsets_buf.typed_data(), NULL,
            columns_buf.typed_data(),
            csr_values_buf.typed_data()), state->status, "update_pointers A");

        CUDSS_CALL_AND_CHECK(cudssMatrixSetValues(state->b, b_values_buf.typed_data()), state->status, "update_pointers b");
        CUDSS_CALL_AND_CHECK(cudssMatrixSetValues(state->x, out_values_buf->typed_data()), state->status, "update_pointers x");

        // warm solve - re-run analysis/factorization/solve. The diagnostic
        // single-solve path is sequence-sensitive with REFACTORIZATION under
        // repeated JAX FFI calls; keep solution-only fast and make diagnostics
        // robust.
        CUDSS_CALL_AND_CHECK(cudssExecute(state->handle, CUDSS_PHASE_ANALYSIS,
            state->config, state->data, state->A, state->x, state->b), state->status, "cudssExecute analysis");

        CUDSS_CALL_AND_CHECK(cudssExecute(state->handle, CUDSS_PHASE_FACTORIZATION,
            state->config, state->data, state->A, state->x, state->b), state->status, "cudssExecute factorization");

        CUDSS_CALL_AND_CHECK(cudssExecute(state->handle, CUDSS_PHASE_SOLVE,
            state->config, state->data, state->A, state->x, state->b), state->status, "cudssExecute solve");
    }

    // Diagnostic extraction - these can fail for certain matrix types (e.g., general non-SPD).
    // Don't fail the solve, just continue with zeros. Query cuDSS inertia directly
    // instead of fetching DIAG/PERM_REORDER_ROW and reconstructing it in JAX.
    // FFI result buffers are device buffers, so copy/zero the inertia result on
    // XLA's stream and synchronize before the stack host buffer goes out of scope.
    int32_t inertia_host[2] = {0, 0};
    state->status = cudssDataGet(state->handle, state->data, CUDSS_DATA_INERTIA,
                    inertia_host, sizeof(inertia_host), &state->sizeWritten);
    if (state->status == CUDSS_STATUS_SUCCESS) {
        CUDA_CHECK(cudaMemcpyAsync(inertia_buf->typed_data(), inertia_host,
                    sizeof(inertia_host), cudaMemcpyHostToDevice, stream));
    } else {
        CUDA_CHECK(cudaMemsetAsync(inertia_buf->typed_data(), 0,
                    sizeof(inertia_host), stream));
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));

    return ffi::Error::Success();
}

template <ffi::DataType T>
static ffi::Error CudssExecuteXOnly(
    cudaStream_t stream,
    CudssState<T>* state,
    ffi::Buffer<T> b_values_buf,
    ffi::Buffer<T> csr_values_buf,
    ffi::Buffer<ffi::S32> offsets_buf,
    ffi::Buffer<ffi::S32> columns_buf,
    ffi::ResultBuffer<T> out_values_buf,
    const int64_t device_id,
    const int64_t mtype_id,
    const int64_t mview_id
) {

    // Track stream for cleanup synchronization
    state->last_stream = stream;

    // instantiate system branch
    if (state->call_count == 0) {

        // figure this out on first call
        state->n = offsets_buf.element_count() - 1;
        state->nnz = columns_buf.element_count();

        // CuDSS setup
        CUDSS_CALL_AND_CHECK(cudssCreate(&state->handle), state->status, "cudssCreate");
        CUDSS_CALL_AND_CHECK(cudssSetStream(state->handle, stream), state->status, "cudssSetStream");
        CUDSS_CALL_AND_CHECK(cudssConfigCreate(&state->config), state->status, "cudssConfigCreate");
        CUDSS_CALL_AND_CHECK(cudssDataCreate(state->handle, &state->data), state->status, "cudssDataCreate");

        // CuDSS structures creation
        CUDSS_CALL_AND_CHECK(cudssMatrixCreateDn(&state->b, state->n, state->nrhs, state->n,
            b_values_buf.typed_data(), state->cuda_dtype, CUDSS_LAYOUT_COL_MAJOR), state->status, "cudssMatrixCreateDn for b");

        CUDSS_CALL_AND_CHECK(cudssMatrixCreateDn(&state->x, state->n, state->nrhs, state->n,
            out_values_buf->typed_data(), state->cuda_dtype, CUDSS_LAYOUT_COL_MAJOR), state->status, "cudssMatrixCreateDn for x");

        CUDSS_CALL_AND_CHECK(cudssMatrixCreateCsr(&state->A, state->n, state->n, state->nnz,
            offsets_buf.typed_data(), NULL,
            columns_buf.typed_data(),
            csr_values_buf.typed_data(),
            CUDA_R_32I, state->cuda_dtype,
            state->mtype, state->mview, state->base), state->status, "cudssMatrixCreateCsr");

        // CuDSS config
        // iterative refinement of the soln is pretty n i f t y
        int iter_ref_nsteps = cudss_ir_nsteps();
        CUDSS_CALL_AND_CHECK(cudssConfigSet(state->config, CUDSS_CONFIG_IR_N_STEPS,
                            &iter_ref_nsteps, sizeof(iter_ref_nsteps)), state->status, "cudssConfigSet ir_nsteps");
        {
            ffi::Error env_error = cudss_apply_env_options_or_error(state->config);
            if (env_error.failure()) return env_error;
        }

        // cold solve - analyze, factorize, solve
        CUDSS_CALL_AND_CHECK(cudssExecute(state->handle, CUDSS_PHASE_ANALYSIS,
            state->config, state->data, state->A, state->x, state->b), state->status, "cudssExecute analysis");

        CUDSS_CALL_AND_CHECK(cudssExecute(state->handle, CUDSS_PHASE_FACTORIZATION,
            state->config, state->data, state->A, state->x, state->b), state->status, "cudssExecute factorization");

        CUDSS_CALL_AND_CHECK(cudssExecute(state->handle, CUDSS_PHASE_SOLVE,
            state->config, state->data, state->A, state->x, state->b), state->status, "cudssExecute solve");

        state->call_count++;
    }
    else {
        // stream can change between calls!!!
        CUDSS_CALL_AND_CHECK(cudssSetStream(state->handle, stream), state->status, "cudssSetStream");

        CUDSS_CALL_AND_CHECK(cudssMatrixSetCsrPointers(state->A,
            offsets_buf.typed_data(), NULL,
            columns_buf.typed_data(),
            csr_values_buf.typed_data()), state->status, "update_pointers A");

        CUDSS_CALL_AND_CHECK(cudssMatrixSetValues(state->b, b_values_buf.typed_data()), state->status, "update_pointers b");
        CUDSS_CALL_AND_CHECK(cudssMatrixSetValues(state->x, out_values_buf->typed_data()), state->status, "update_pointers x");

        // warm solve - refactorize, solve
        CUDSS_CALL_AND_CHECK(cudssExecute(state->handle, CUDSS_PHASE_REFACTORIZATION,
            state->config, state->data, state->A, state->x, state->b), state->status, "cudssExecute refactorization");

        CUDSS_CALL_AND_CHECK(cudssExecute(state->handle, CUDSS_PHASE_SOLVE,
            state->config, state->data, state->A, state->x, state->b), state->status, "cudssExecute solve");
    }

    return ffi::Error::Success();
}

template <ffi::DataType T>
static ffi::Error CudssExecuteConstantXOnly(
    cudaStream_t stream,
    CudssState<T>* state,
    ffi::Buffer<T> b_values_buf,
    ffi::Buffer<T> csr_values_buf,
    ffi::Buffer<ffi::S32> offsets_buf,
    ffi::Buffer<ffi::S32> columns_buf,
    ffi::ResultBuffer<T> out_values_buf,
    const int64_t device_id,
    const int64_t mtype_id,
    const int64_t mview_id
) {
    std::lock_guard<std::mutex> invocation_lock(state->constant_mutex);
    CUDA_CHECK(cudaSetDevice(static_cast<int>(state->device_id)));
    state->last_stream = stream;
    bool stream_completed = false;
    struct StreamCompletionGuard {
        cudaStream_t stream;
        bool& completed;
        ~StreamCompletionGuard() { if (!completed && stream) cudaStreamSynchronize(stream); }
    } completion_guard{stream, stream_completed};

    const int64_t n = offsets_buf.element_count() - 1;
    const int64_t nnz = columns_buf.element_count();
    if (n <= 0 || b_values_buf.element_count() != n ||
        csr_values_buf.element_count() != nnz ||
        offsets_buf.element_count() != n + 1) {
        return ffi::Error::Internal("invalid constant single-RHS CSR/RHS dimensions");
    }

    const bool cold = state->call_count == 0;
    struct ColdRollback {
        CudssState<T>* state;
        bool active;
        ~ColdRollback() { if (active) state->DestroyResources(); }
    } rollback{state, cold};

    if (cold) {
        state->n = n;
        state->nnz = nnz;
        state->nrhs = 1;
        state->owned_csr_offsets_bytes = (n + 1) * sizeof(int32_t);
        state->owned_csr_columns_bytes = nnz * sizeof(int32_t);
        state->owned_csr_values_bytes = nnz * sizeof(typename CudssState<T>::native_dtype);
        CUDA_CHECK(cudaMallocAsync(reinterpret_cast<void**>(&state->owned_csr_offsets), state->owned_csr_offsets_bytes, stream));
        CUDA_CHECK(cudaMallocAsync(reinterpret_cast<void**>(&state->owned_csr_columns), state->owned_csr_columns_bytes, stream));
        CUDA_CHECK(cudaMallocAsync(reinterpret_cast<void**>(&state->owned_csr_values), state->owned_csr_values_bytes, stream));
        CUDA_CHECK(cudaMemcpyAsync(state->owned_csr_offsets, offsets_buf.typed_data(), state->owned_csr_offsets_bytes, cudaMemcpyDeviceToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(state->owned_csr_columns, columns_buf.typed_data(), state->owned_csr_columns_bytes, cudaMemcpyDeviceToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(state->owned_csr_values, csr_values_buf.typed_data(), state->owned_csr_values_bytes, cudaMemcpyDeviceToDevice, stream));

        CUDSS_CALL_AND_CHECK(cudssCreate(&state->handle), state->status, "cudssCreate");
        CUDSS_CALL_AND_CHECK(cudssSetStream(state->handle, stream), state->status, "cudssSetStream");
        CUDSS_CALL_AND_CHECK(cudssConfigCreate(&state->config), state->status, "cudssConfigCreate");
        CUDSS_CALL_AND_CHECK(cudssDataCreate(state->handle, &state->data), state->status, "cudssDataCreate");

        CUDSS_CALL_AND_CHECK(cudssMatrixCreateDn(&state->b, state->n, state->nrhs, state->n,
            b_values_buf.typed_data(), state->cuda_dtype, CUDSS_LAYOUT_COL_MAJOR), state->status, "cudssMatrixCreateDn for b");
        CUDSS_CALL_AND_CHECK(cudssMatrixCreateDn(&state->x, state->n, state->nrhs, state->n,
            out_values_buf->typed_data(), state->cuda_dtype, CUDSS_LAYOUT_COL_MAJOR), state->status, "cudssMatrixCreateDn for x");
        CUDSS_CALL_AND_CHECK(cudssMatrixCreateCsr(&state->A, state->n, state->n, state->nnz,
            state->owned_csr_offsets, NULL,
            state->owned_csr_columns,
            state->owned_csr_values,
            CUDA_R_32I, state->cuda_dtype,
            state->mtype, state->mview, state->base), state->status, "cudssMatrixCreateCsr");

        int iter_ref_nsteps = cudss_ir_nsteps();
        CUDSS_CALL_AND_CHECK(cudssConfigSet(state->config, CUDSS_CONFIG_IR_N_STEPS,
                            &iter_ref_nsteps, sizeof(iter_ref_nsteps)), state->status, "cudssConfigSet ir_nsteps");
        {
            ffi::Error env_error = cudss_apply_env_options_or_error(state->config);
            if (env_error.failure()) return env_error;
        }

        CUDSS_CALL_AND_CHECK(cudssExecute(state->handle, CUDSS_PHASE_ANALYSIS,
            state->config, state->data, state->A, state->x, state->b), state->status, "cudssExecute analysis");
        CUDSS_CALL_AND_CHECK(cudssExecute(state->handle, CUDSS_PHASE_FACTORIZATION,
            state->config, state->data, state->A, state->x, state->b), state->status, "cudssExecute factorization");
        CUDSS_CALL_AND_CHECK(cudssExecute(state->handle, CUDSS_PHASE_SOLVE,
            state->config, state->data, state->A, state->x, state->b), state->status, "cudssExecute solve");
        state->call_count++;
        rollback.active = false;
    } else {
        if (state->n != n || state->nnz != nnz || state->nrhs != 1) {
            return ffi::Error::Internal("constant single-RHS cuDSS state called with changed shape");
        }
        CUDSS_CALL_AND_CHECK(cudssSetStream(state->handle, stream), state->status, "cudssSetStream");
        CUDSS_CALL_AND_CHECK(cudssMatrixSetValues(state->b, b_values_buf.typed_data()), state->status, "update_pointers b");
        CUDSS_CALL_AND_CHECK(cudssMatrixSetValues(state->x, out_values_buf->typed_data()), state->status, "update_pointers x");
        CUDSS_CALL_AND_CHECK(cudssExecute(state->handle, CUDSS_PHASE_SOLVE,
            state->config, state->data, state->A, state->x, state->b), state->status, "cudssExecute solve");
    }

    CUDA_CHECK(cudaStreamSynchronize(stream));
    stream_completed = true;
    return ffi::Error::Success();
}

/* Constant-matrix multi-RHS x-only solve.
 *
 * b_values_buf/out_values_buf are logical JAX arrays of shape (nrhs, n). JAX's
 * default row-major physical layout stores each RHS row contiguously. cuDSS is
 * given the same pointer as a column-major dense matrix with n rows, nrhs
 * columns, and ld=n, so each contiguous row becomes one RHS column. This avoids
 * explicit transposes/copies while solving A X = B with one factorization.
 */
template <ffi::DataType T>
static ffi::Error CudssExecuteConstantMultiRHSXOnly(
    cudaStream_t stream,
    CudssState<T>* state,
    ffi::Buffer<T> b_values_buf,
    ffi::Buffer<T> csr_values_buf,
    ffi::Buffer<ffi::S32> offsets_buf,
    ffi::Buffer<ffi::S32> columns_buf,
    ffi::ResultBuffer<T> out_values_buf,
    const int64_t device_id,
    const int64_t mtype_id,
    const int64_t mview_id
) {
    std::lock_guard<std::mutex> invocation_lock(state->constant_mutex);
    CUDA_CHECK(cudaSetDevice(static_cast<int>(state->device_id)));
    state->last_stream = stream;
    bool stream_completed = false;
    struct StreamCompletionGuard {
        cudaStream_t stream;
        bool& completed;
        ~StreamCompletionGuard() { if (!completed && stream) cudaStreamSynchronize(stream); }
    } completion_guard{stream, stream_completed};

    const int64_t n = offsets_buf.element_count() - 1;
    const int64_t nnz = columns_buf.element_count();
    if (n <= 0 || b_values_buf.element_count() % n != 0 ||
        csr_values_buf.element_count() != nnz) {
        return ffi::Error::Internal("invalid constant multi-RHS CSR/RHS dimensions");
    }
    const int64_t nrhs = b_values_buf.element_count() / n;
    const bool cold = state->call_count == 0;
    struct ColdRollback {
        CudssState<T>* state;
        bool active;
        ~ColdRollback() { if (active) state->DestroyResources(); }
    } rollback{state, cold};

    if (cold) {
        state->n = n;
        state->nnz = nnz;
        state->nrhs = nrhs;
        state->owned_csr_offsets_bytes = (n + 1) * sizeof(int32_t);
        state->owned_csr_columns_bytes = nnz * sizeof(int32_t);
        state->owned_csr_values_bytes = nnz * sizeof(typename CudssState<T>::native_dtype);
        CUDA_CHECK(cudaMallocAsync(reinterpret_cast<void**>(&state->owned_csr_offsets), state->owned_csr_offsets_bytes, stream));
        CUDA_CHECK(cudaMallocAsync(reinterpret_cast<void**>(&state->owned_csr_columns), state->owned_csr_columns_bytes, stream));
        CUDA_CHECK(cudaMallocAsync(reinterpret_cast<void**>(&state->owned_csr_values), state->owned_csr_values_bytes, stream));
        CUDA_CHECK(cudaMemcpyAsync(state->owned_csr_offsets, offsets_buf.typed_data(), state->owned_csr_offsets_bytes, cudaMemcpyDeviceToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(state->owned_csr_columns, columns_buf.typed_data(), state->owned_csr_columns_bytes, cudaMemcpyDeviceToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(state->owned_csr_values, csr_values_buf.typed_data(), state->owned_csr_values_bytes, cudaMemcpyDeviceToDevice, stream));

        CUDSS_CALL_AND_CHECK(cudssCreate(&state->handle), state->status, "cudssCreate");
        CUDSS_CALL_AND_CHECK(cudssSetStream(state->handle, stream), state->status, "cudssSetStream");
        CUDSS_CALL_AND_CHECK(cudssConfigCreate(&state->config), state->status, "cudssConfigCreate");
        CUDSS_CALL_AND_CHECK(cudssDataCreate(state->handle, &state->data), state->status, "cudssDataCreate");

        CUDSS_CALL_AND_CHECK(cudssMatrixCreateDn(&state->b, state->n, state->nrhs, state->n,
            b_values_buf.typed_data(), state->cuda_dtype, CUDSS_LAYOUT_COL_MAJOR), state->status, "cudssMatrixCreateDn for multi-RHS b");
        CUDSS_CALL_AND_CHECK(cudssMatrixCreateDn(&state->x, state->n, state->nrhs, state->n,
            out_values_buf->typed_data(), state->cuda_dtype, CUDSS_LAYOUT_COL_MAJOR), state->status, "cudssMatrixCreateDn for multi-RHS x");
        CUDSS_CALL_AND_CHECK(cudssMatrixCreateCsr(&state->A, state->n, state->n, state->nnz,
            state->owned_csr_offsets, NULL,
            state->owned_csr_columns,
            state->owned_csr_values,
            CUDA_R_32I, state->cuda_dtype,
            state->mtype, state->mview, state->base), state->status, "cudssMatrixCreateCsr");

        int iter_ref_nsteps = cudss_ir_nsteps();
        CUDSS_CALL_AND_CHECK(cudssConfigSet(state->config, CUDSS_CONFIG_IR_N_STEPS,
                            &iter_ref_nsteps, sizeof(iter_ref_nsteps)), state->status, "cudssConfigSet ir_nsteps");
        {
            ffi::Error env_error = cudss_apply_env_options_or_error(state->config);
            if (env_error.failure()) return env_error;
        }

        CUDSS_CALL_AND_CHECK(cudssExecute(state->handle, CUDSS_PHASE_ANALYSIS,
            state->config, state->data, state->A, state->x, state->b), state->status, "cudssExecute analysis");
        CUDSS_CALL_AND_CHECK(cudssExecute(state->handle, CUDSS_PHASE_FACTORIZATION,
            state->config, state->data, state->A, state->x, state->b), state->status, "cudssExecute factorization");
        CUDSS_CALL_AND_CHECK(cudssExecute(state->handle, CUDSS_PHASE_SOLVE,
            state->config, state->data, state->A, state->x, state->b), state->status, "cudssExecute solve");
        state->call_count++;
        rollback.active = false;
    } else {
        if (state->n != n || state->nnz != nnz || state->nrhs != nrhs) {
            return ffi::Error::Internal("constant multi-RHS cuDSS state called with changed shape");
        }
        CUDSS_CALL_AND_CHECK(cudssSetStream(state->handle, stream), state->status, "cudssSetStream");
        CUDSS_CALL_AND_CHECK(cudssMatrixSetValues(state->b, b_values_buf.typed_data()), state->status, "update_pointers b");
        CUDSS_CALL_AND_CHECK(cudssMatrixSetValues(state->x, out_values_buf->typed_data()), state->status, "update_pointers x");
        CUDSS_CALL_AND_CHECK(cudssExecute(state->handle, CUDSS_PHASE_SOLVE,
            state->config, state->data, state->A, state->x, state->b), state->status, "cudssExecute solve");
    }

    CUDA_CHECK(cudaStreamSynchronize(stream));
    stream_completed = true;
    return ffi::Error::Success();
}

// minimize XLA/nanobind boilerplate with a couple macros ======================

// XLA ffi handler definitions for all datatypes
#define DEFINE_CUDSS_FFI_HANDLERS(TypeName, DataType) \
    XLA_FFI_DEFINE_HANDLER(kCudssInstantiate##TypeName, CudssInstantiate<DataType>, \
        ffi::Ffi::BindInstantiate() \
            .Attr<int64_t>("device_id") \
            .Attr<int64_t>("mtype_id") \
            .Attr<int64_t>("mview_id")); \
    \
    XLA_FFI_DEFINE_HANDLER(kCudssExecute##TypeName, CudssExecute<DataType>, \
        ffi::Ffi::Bind() \
            .Ctx<ffi::PlatformStream<cudaStream_t>>() \
            .Ctx<ffi::State<CudssState<DataType>>>() \
            .Arg<ffi::Buffer<DataType>>() \
            .Arg<ffi::Buffer<DataType>>() \
            .Arg<ffi::Buffer<ffi::S32>>() \
            .Arg<ffi::Buffer<ffi::S32>>() \
            .Ret<ffi::Buffer<DataType>>() \
            .Ret<ffi::Buffer<ffi::S32>>() \
            .Attr<int64_t>("device_id") \
            .Attr<int64_t>("mtype_id") \
            .Attr<int64_t>("mview_id")); \
    \
    XLA_FFI_DEFINE_HANDLER(kCudssExecuteXOnly##TypeName, CudssExecuteXOnly<DataType>, \
        ffi::Ffi::Bind() \
            .Ctx<ffi::PlatformStream<cudaStream_t>>() \
            .Ctx<ffi::State<CudssState<DataType>>>() \
            .Arg<ffi::Buffer<DataType>>() \
            .Arg<ffi::Buffer<DataType>>() \
            .Arg<ffi::Buffer<ffi::S32>>() \
            .Arg<ffi::Buffer<ffi::S32>>() \
            .Ret<ffi::Buffer<DataType>>() \
            .Attr<int64_t>("device_id") \
            .Attr<int64_t>("mtype_id") \
            .Attr<int64_t>("mview_id")); \
    \
    XLA_FFI_DEFINE_HANDLER(kCudssExecuteConstantXOnly##TypeName, CudssExecuteConstantXOnly<DataType>, \
        ffi::Ffi::Bind() \
            .Ctx<ffi::PlatformStream<cudaStream_t>>() \
            .Ctx<ffi::State<CudssState<DataType>>>() \
            .Arg<ffi::Buffer<DataType>>() \
            .Arg<ffi::Buffer<DataType>>() \
            .Arg<ffi::Buffer<ffi::S32>>() \
            .Arg<ffi::Buffer<ffi::S32>>() \
            .Ret<ffi::Buffer<DataType>>() \
            .Attr<int64_t>("device_id") \
            .Attr<int64_t>("mtype_id") \
            .Attr<int64_t>("mview_id")); \
    \
    XLA_FFI_DEFINE_HANDLER(kCudssExecuteConstantMultiRHSXOnly##TypeName, CudssExecuteConstantMultiRHSXOnly<DataType>, \
        ffi::Ffi::Bind() \
            .Ctx<ffi::PlatformStream<cudaStream_t>>() \
            .Ctx<ffi::State<CudssState<DataType>>>() \
            .Arg<ffi::Buffer<DataType>>() \
            .Arg<ffi::Buffer<DataType>>() \
            .Arg<ffi::Buffer<ffi::S32>>() \
            .Arg<ffi::Buffer<ffi::S32>>() \
            .Ret<ffi::Buffer<DataType>>() \
            .Attr<int64_t>("device_id") \
            .Attr<int64_t>("mtype_id") \
            .Attr<int64_t>("mview_id"));

// Generate all the FFI handlers using the macro
DEFINE_CUDSS_FFI_HANDLERS(f32, ffi::F32);
DEFINE_CUDSS_FFI_HANDLERS(f64, ffi::F64);
DEFINE_CUDSS_FFI_HANDLERS(c64, ffi::C64);
DEFINE_CUDSS_FFI_HANDLERS(c128, ffi::C128);

#if defined(XLA_FFI_API_MINOR) && (XLA_FFI_API_MINOR >= 2)
  #define ADD_TYPE(d, DTYPE) do { \
      using StateT = CudssState<DTYPE>; \
      static auto kStateTypeInfo = xla::ffi::MakeTypeInfo<StateT>(); \
      (d)["type_info"] = nb::capsule(reinterpret_cast<void*>(&kStateTypeInfo)); \
      (d)["type_id"]   = nb::capsule(reinterpret_cast<void*>(&StateT::id)); \
    } while (0)
#else
  #define ADD_TYPE(d, DTYPE) do { \
      (d)["state_type"] = nb::dict(); \
    } while (0)
#endif

// nanobind module exporting macro
#define EXPORT_CUDSS_HANDLERS(m, TypeName, DataType) \
    m.def("state_dict_" #TypeName, []() { \
        nb::dict d; \
        ADD_TYPE(d, DataType); \
        return d; \
    }); \
    m.def("type_id_" #TypeName, []() { \
        return nb::capsule(reinterpret_cast<void*>(&CudssState<DataType>::id)); \
    }); \
    m.def("handler_" #TypeName, []() { \
        nb::dict d; \
        d["instantiate"] = nb::capsule(reinterpret_cast<void*>(kCudssInstantiate##TypeName)); \
        d["execute"] = nb::capsule(reinterpret_cast<void*>(kCudssExecute##TypeName)); \
        return d; \
    }); \
    m.def("handler_xonly_" #TypeName, []() { \
        nb::dict d; \
        d["instantiate"] = nb::capsule(reinterpret_cast<void*>(kCudssInstantiate##TypeName)); \
        d["execute"] = nb::capsule(reinterpret_cast<void*>(kCudssExecuteXOnly##TypeName)); \
        return d; \
    }); \
    m.def("handler_const_xonly_" #TypeName, []() { \
        nb::dict d; \
        d["instantiate"] = nb::capsule(reinterpret_cast<void*>(kCudssInstantiate##TypeName)); \
        d["execute"] = nb::capsule(reinterpret_cast<void*>(kCudssExecuteConstantXOnly##TypeName)); \
        return d; \
    }); \
    m.def("handler_const_multi_rhs_xonly_" #TypeName, []() { \
        nb::dict d; \
        d["instantiate"] = nb::capsule(reinterpret_cast<void*>(kCudssInstantiate##TypeName)); \
        d["execute"] = nb::capsule(reinterpret_cast<void*>(kCudssExecuteConstantMultiRHSXOnly##TypeName)); \
        return d; \
    });

// generate all nanobind modules! :)
NB_MODULE(single_solve, m) {
    EXPORT_CUDSS_HANDLERS(m, f32, ffi::F32);
    EXPORT_CUDSS_HANDLERS(m, f64, ffi::F64);
    EXPORT_CUDSS_HANDLERS(m, c64, ffi::C64);
    EXPORT_CUDSS_HANDLERS(m, c128, ffi::C128);
}
