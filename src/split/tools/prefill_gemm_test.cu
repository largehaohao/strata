// The same native FP32 WMMA entry point used by the split prompt path.
#include "wmma_gemm.h"
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>
#include <hipblas/hipblas.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <stdexcept>
#include <vector>

static void ck(hipError_t e) {
    if (e != hipSuccess) throw std::runtime_error(hipGetErrorString(e));
}
static void cb(hipblasStatus_t e) {
    if (e != HIPBLAS_STATUS_SUCCESS) throw std::runtime_error("hipBLAS failed");
}
struct Buffer {
    void* p = nullptr;
    explicit Buffer(size_t n) { ck(hipMalloc(&p, n)); }
    ~Buffer() { if (p) (void) hipFree(p); }
};
struct Shape { int t, n, k, pad; };
static bool native(const uint16_t* x,const uint16_t* w,float* y,Shape s,hipStream_t stream,int ld=0) {
    if(!ld)ld=s.k;
    return strata_pf_gemm_f16_ld(x,ld,w,ld,y,s.t,s.n,s.k,s.n+s.pad,0,stream) ||
           strata_pf_gemm_narrow_f16_ld(x,ld,w,ld,y,s.t,s.n,s.k,s.n+s.pad,stream);
}

static void check_case(Shape s, int mode, hipStream_t stream, hipblasHandle_t blas) {
    const int ld = s.n + s.pad;
    const bool padded=mode>=3;
    const int values=mode%3,input_ld=s.k+(padded?64:0);
    std::mt19937 rng(773 + s.t + s.n + values);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    std::vector<__half> x((size_t)s.t * s.k), w((size_t)s.n * s.k);
    auto value = [&] {
        return __float2half(values == 2 ? 64.0f : values == 0 ? (int(rng() % 17) - 8) / 16.0f : dist(rng));
    };
    for (auto& v : x) v = value();
    for (auto& v : w) v = value();
    constexpr float guard = -8888.25f;
    std::vector<float> y((size_t)s.t * ld + 16, guard);
    Buffer dx((size_t)s.t*input_ld*2), dw((size_t)s.n*input_ld*2), dy(y.size()*4);
    ck(hipMemsetAsync(dx.p,0x7f,(size_t)s.t*input_ld*2,stream));
    ck(hipMemsetAsync(dw.p,0x7f,(size_t)s.n*input_ld*2,stream));
    ck(hipMemcpy2DAsync(dx.p,input_ld*2,x.data(),s.k*2,s.k*2,s.t,hipMemcpyHostToDevice,stream));
    ck(hipMemcpy2DAsync(dw.p,input_ld*2,w.data(),s.k*2,s.k*2,s.n,hipMemcpyHostToDevice,stream));
    ck(hipMemcpyAsync(dy.p,y.data(),y.size()*4,hipMemcpyHostToDevice,stream));
    if (!native((uint16_t*)dx.p,(uint16_t*)dw.p,(float*)dy.p,s,stream,input_ld))
        throw std::runtime_error("expected WMMA dispatch");
    ck(hipMemcpyAsync(y.data(),dy.p,y.size()*4,hipMemcpyDeviceToHost,stream));
    ck(hipStreamSynchronize(stream));
    if(padded) {
        Buffer cx(x.size()*2),cw(w.size()*2),cy(y.size()*4);
        std::vector<float> reference(y.size(),guard);
        ck(hipMemcpyAsync(cx.p,x.data(),x.size()*2,hipMemcpyHostToDevice,stream));
        ck(hipMemcpyAsync(cw.p,w.data(),w.size()*2,hipMemcpyHostToDevice,stream));
        ck(hipMemcpyAsync(cy.p,reference.data(),reference.size()*4,hipMemcpyHostToDevice,stream));
        if(!native((uint16_t*)cx.p,(uint16_t*)cw.p,(float*)cy.p,s,stream)) throw std::runtime_error("compact control dispatch");
        ck(hipMemcpyAsync(reference.data(),cy.p,reference.size()*4,hipMemcpyDeviceToHost,stream));ck(hipStreamSynchronize(stream));
        if(std::memcmp(reference.data(),y.data(),y.size()*4)) throw std::runtime_error("padded GEMM differs from compact GEMM");
    }
    if (s.n == 320) {
        // Independent dispatch of the original 128-column kernel. Zero-pad
        // only W's output rows; the actual dot products and K order are intact.
        Buffer wide_w((size_t)512*input_ld*2), wide_y((size_t)s.t*512*4);
        ck(hipMemsetAsync(wide_w.p,0,(size_t)512*input_ld*2,stream));
        ck(hipMemcpyAsync(wide_w.p,dw.p,(size_t)s.n*input_ld*2,hipMemcpyDeviceToDevice,stream));
        if(!strata_pf_gemm_f16_ld((uint16_t*)dx.p,input_ld,(uint16_t*)wide_w.p,input_ld,
            (float*)wide_y.p,s.t,512,s.k,512,0,stream)) throw std::runtime_error("narrow control dispatch");
        std::vector<float> wide((size_t)s.t*512);
        ck(hipMemcpyAsync(wide.data(),wide_y.p,wide.size()*4,hipMemcpyDeviceToHost,stream));
        ck(hipStreamSynchronize(stream));
        for(int t=0;t<s.t;++t)
            if(std::memcmp(wide.data()+(size_t)t*512,y.data()+(size_t)t*ld,s.n*4))
                throw std::runtime_error("narrow tile changed FP32 product bits");
    }
    for (int t=0;t<s.t;++t) {
        for (int n=0;n<s.n;++n) if (!std::isfinite(y[(size_t)t*ld+n]))
            throw std::runtime_error("nonfinite FP32 product");
        for (int n=s.n;n<ld;++n) if (y[(size_t)t*ld+n]!=guard)
            throw std::runtime_error("output row padding overwritten");
    }
    for (size_t i=(size_t)s.t*ld;i<y.size();++i) if (y[i]!=guard)
        throw std::runtime_error("output end guard overwritten");
    const size_t cells=(size_t)s.t*s.n;
    const bool full=cells*s.k<10000000;
    const size_t samples=full?cells:257;
    // Keep a BLAS result as a control for the precision comparison.
    const float one_ref=1,zero_ref=0;
    cb(hipblasGemmEx(blas,HIPBLAS_OP_T,HIPBLAS_OP_N,s.n,s.t,s.k,&one_ref,
        dw.p,HIPBLAS_R_16F,input_ld,dx.p,HIPBLAS_R_16F,input_ld,&zero_ref,dy.p,HIPBLAS_R_32F,ld,
        HIPBLAS_COMPUTE_32F,HIPBLAS_GEMM_DEFAULT));
    std::vector<float> by(y.size());
    ck(hipMemcpyAsync(by.data(),dy.p,by.size()*4,hipMemcpyDeviceToHost,stream));ck(hipStreamSynchronize(stream));
    double worst=0, sqerr=0, sqref=0, worst_scaled=0, blas_worst=0, blas_sq=0;
    for(size_t i=0;i<samples;++i) {
        const size_t cell=full?i:(i==0?0:i==1?cells-1:(i*104729+31)%cells);
        const int t=cell/s.n,n=cell%s.n;
        double ref=0, magnitude=0;
        for(int k=0;k<s.k;++k) {
            const double term=(double)__half2float(x[(size_t)t*s.k+k])*__half2float(w[(size_t)n*s.k+k]);
            ref+=term;magnitude+=std::abs(term);
        }
        const double error=std::abs((double)y[(size_t)t*ld+n]-ref);
        worst=std::max(worst,error);sqerr+=error*error;sqref+=ref*ref;
        const double be=std::abs((double)by[(size_t)t*ld+n]-ref);
        blas_worst=std::max(blas_worst,be);blas_sq+=be*be;
        worst_scaled=std::max(worst_scaled,error/std::max(1.0,magnitude));
        // Use the actual engine self-test's gfx11 WMMA tolerance. Long random
        // dot products accumulate more rounding than binary-fraction fixtures.
        // Also require relative L2 <= 1e-4 below, independently of cancellation.
        const double tolerance=1e-4*std::max(1.0,magnitude);
        if (error>tolerance) {
            std::fprintf(stderr,"mismatch T=%d N=%d K=%d t=%d n=%d got=%.9g ref=%.12g error=%.8g\n",s.t,s.n,s.k,t,n,y[(size_t)t*ld+n],ref,error);
            throw std::runtime_error("FP64 oracle mismatch");
        }
    }
    std::printf("oracle T=%d N=%d K=%d mode=%d: WMMA abs %.4g scaled %.4g L2 %.4g; BLAS abs %.4g L2 %.4g\n",
        s.t,s.n,s.k,mode,worst,worst_scaled,std::sqrt(sqerr/std::max(1.0,sqref)),blas_worst,std::sqrt(blas_sq/std::max(1.0,sqref)));
    if(std::sqrt(sqerr/std::max(1.0,sqref))>0.0001)
        throw std::runtime_error("relative L2 error");
    // Actual split layout, strides and FP32 output, timed on the same stream.
    if(values==1) {
        hipEvent_t a,b;ck(hipEventCreate(&a));ck(hipEventCreate(&b));
        float ms[2]={};const float one=1,zero=0;
        cb(hipblasGemmEx(blas,HIPBLAS_OP_T,HIPBLAS_OP_N,s.n,s.t,s.k,&one,
            dw.p,HIPBLAS_R_16F,input_ld,dx.p,HIPBLAS_R_16F,input_ld,&zero,dy.p,HIPBLAS_R_32F,ld,
            HIPBLAS_COMPUTE_32F,HIPBLAS_GEMM_DEFAULT));
        ck(hipStreamSynchronize(stream));
        for(int which=0;which<2;++which) {
            ck(hipEventRecord(a,stream));
            for(int i=0;i<5;++i) {
                if(which==0) {
                    if(!native((uint16_t*)dx.p,(uint16_t*)dw.p,(float*)dy.p,s,stream,input_ld))
                        throw std::runtime_error("WMMA benchmark dispatch");
                } else cb(hipblasGemmEx(blas,HIPBLAS_OP_T,HIPBLAS_OP_N,s.n,s.t,s.k,&one,
                    dw.p,HIPBLAS_R_16F,input_ld,dx.p,HIPBLAS_R_16F,input_ld,&zero,dy.p,HIPBLAS_R_32F,ld,
                    HIPBLAS_COMPUTE_32F,HIPBLAS_GEMM_DEFAULT));
            }
            ck(hipEventRecord(b,stream));ck(hipEventSynchronize(b));ck(hipEventElapsedTime(&ms[which],a,b));
        }
        ck(hipEventDestroy(a));ck(hipEventDestroy(b));
        std::printf("T=%d N=%d K=%d ldy=%d input_ld=%d: WMMA %.3f ms BLAS %.3f ms max_abs %.3g\n",s.t,s.n,s.k,ld,input_ld,ms[0]/5,ms[1]/5,worst);
    }
}

// The original head projection zero-fills a full-model slice, runs GEMM with
// that output stride, then gathers the local heads. Compare its actual slice
// boundaries against compact products, including a shard crossing a slice.
static int check_gsu3(int n, int k, hipStream_t stream, hipblasHandle_t blas) {
    constexpr int max_t = 63;
    const int max_ld = n + 9;
    const auto dispatch = n == 320 ? strata_pf_hcdown_gsu3_f16 : strata_pf_wide_gsu3_f16;
    Buffer x((size_t)max_t * k * 2), w((size_t)n * k * 2), y(((size_t)max_t * max_ld + 17) * 4);
    std::vector<__half> hx((size_t)max_t * k), hw((size_t)n * k);
    std::mt19937 rng(597);
    std::uniform_real_distribution<float> uniform(-1.f, 1.f);
    int checks = 0;
    for (bool mixed_magnitudes : {false, true}) {
        auto value = [&]() {
            const float v = uniform(rng);
            return __float2half(mixed_magnitudes ? std::ldexp(v, (int)(rng() % 33) - 21) : v);
        };
        std::generate(hx.begin(), hx.end(), value);
        std::generate(hw.begin(), hw.end(), value);
        ck(hipMemcpy(x.p, hx.data(), hx.size()*2, hipMemcpyHostToDevice));
        ck(hipMemcpy(w.p, hw.data(), hw.size()*2, hipMemcpyHostToDevice));
        for (int t = 16; t <= max_t; ++t) for (int ld : {n, max_ld}) {
            const size_t size = (size_t)t * ld + 17;
            std::vector<uint32_t> reference(size), actual(size);
            const float one = 1.f, zero = 0.f;
            ck(hipMemsetAsync(y.p, 0xa5, size*4, stream));
            cb(hipblasGemmEx(blas, HIPBLAS_OP_T, HIPBLAS_OP_N, n, t, k, &one, w.p, HIPBLAS_R_16F, k,
                x.p, HIPBLAS_R_16F, k, &zero, y.p, HIPBLAS_R_32F, ld, HIPBLAS_COMPUTE_32F, HIPBLAS_GEMM_DEFAULT));
            ck(hipMemcpyAsync(reference.data(), y.p, size*4, hipMemcpyDeviceToHost, stream));
            ck(hipMemsetAsync(y.p, 0xa5, size*4, stream));
            if (!dispatch((uint16_t*)x.p, (uint16_t*)w.p, (float*)y.p, t, n, k, ld, stream))
                throw std::runtime_error("GSU3 dispatch");
            ck(hipMemcpyAsync(actual.data(), y.p, size*4, hipMemcpyDeviceToHost, stream));
            ck(hipStreamSynchronize(stream));
            for (size_t i=0; i<size; ++i) {
                const bool padding = i >= (size_t)t*ld || i%ld >= n;
                if (actual[i] != reference[i] || (padding && actual[i] != 0xa5a5a5a5u)) {
                    std::fprintf(stderr, "GSU3 mismatch at T=%d N=%d K=%d ld=%d cell=%zu mixed=%d ref=%08x actual=%08x\n",
                                 t, n, k, ld, i, mixed_magnitudes, reference[i], actual[i]);
                    throw std::runtime_error("GSU3 FP32 equality or padding");
                }
            }
            ++checks;
        }
    }
    for (const Shape s : {Shape{15,n,k,0}, {64,n,k,0}, {19,n-1,k,0}, {19,n,k-16,0}, {19,n,k,-1}}) {
        if (dispatch((uint16_t*)x.p,(uint16_t*)w.p,(float*)y.p,s.t,s.n,s.k,s.n+s.pad,stream))
            throw std::runtime_error("unsupported GSU3 shape accepted");
        ++checks;
    }
    return checks;
}

// Reference the production dequantization, including the padding of a head shard.
__global__ void expand_q8(const int8_t* bytes, const uint16_t* scales, __half* out,
                          size_t elements, int k, int first, int owned) {
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < elements; i += gridDim.x * blockDim.x) {
        const size_t col = i / k;
        const float value = col >= first && col < first + owned
            ? __half2float(__ushort_as_half(scales[i / 32])) * bytes[i] : 0.f;
        out[i] = __float2half(value);
    }
}

static int check_q8_gsu3(int n, int k, int first, int owned, hipStream_t stream) {
    constexpr int max_t = 63, pad = 9, guard = 17;
    constexpr uint32_t guard_bits = 0xa5a5a5a5u;
    const size_t weights = (size_t)n * k;
    Buffer x((size_t)max_t * k * sizeof(__half)), q(weights), scales(weights / 32 * sizeof(uint16_t));
    Buffer w(weights * sizeof(__half)), y(((size_t)max_t * (n + pad) + guard) * sizeof(float));
    std::vector<__half> hx((size_t)max_t * k);
    std::vector<int8_t> hq(weights);
    std::vector<uint16_t> hs(weights / 32);
    std::mt19937 rng(407 + n + first);
    for (auto& value : hx) value = __float2half((int(rng() % 2049) - 1024) * 0x1p-10f);
    for (auto& value : hq) value = (int)(rng() % 256) - 128;
    constexpr uint16_t scale_bits[] = {0, 1, 3, 0x200, 0x3ff, 0x400, 0x1400, 0x2400, 0x3800};
    for (size_t i = 0; i < hs.size(); ++i) hs[i] = scale_bits[i % 9] | (i % 11 == 0 ? 0x8000 : 0);
    ck(hipMemcpy(x.p, hx.data(), hx.size() * sizeof(__half), hipMemcpyHostToDevice));
    ck(hipMemcpy(q.p, hq.data(), hq.size(), hipMemcpyHostToDevice));
    ck(hipMemcpy(scales.p, hs.data(), hs.size() * sizeof(uint16_t), hipMemcpyHostToDevice));
    expand_q8<<<1024, 256, 0, stream>>>(static_cast<int8_t*>(q.p), static_cast<uint16_t*>(scales.p),
        static_cast<__half*>(w.p), weights, k, first, owned);
    ck(hipGetLastError());

    const auto ref = n == 320 ? strata_pf_hcdown_gsu3_f16 : strata_pf_wide_gsu3_f16;
    const auto* input = static_cast<const uint16_t*>(x.p);
    const auto* bytes = static_cast<const int8_t*>(q.p) + (size_t)first * k;
    const auto* scale = static_cast<const uint16_t*>(scales.p) + (size_t)first * k / 32;
    auto* output = static_cast<float*>(y.p);
    int checks = 0;
    for (int rows : {16, 19, 27, max_t}) for (int ld : {n, n + pad}) {
        const size_t cells = (size_t)rows * ld + guard;
        const size_t size = cells * sizeof(float);
        std::vector<uint32_t> expected(cells), actual(cells);
        ck(hipMemsetAsync(output, 0xa5, size, stream));
        if (!ref(input, static_cast<const uint16_t*>(w.p), output, rows, n, k, ld, stream))
            throw std::runtime_error("Q8 reference dispatch");
        ck(hipMemcpyAsync(expected.data(), output, size, hipMemcpyDeviceToHost, stream));
        ck(hipMemsetAsync(output, 0xa5, size, stream));
        if (!strata_pf_q8_gsu3_f16(input, bytes, scale, output + first, rows, owned, k, ld, n, first, stream))
            throw std::runtime_error("Q8 GSU3 dispatch");
        ck(hipMemcpyAsync(actual.data(), output, size, hipMemcpyDeviceToHost, stream));
        ck(hipStreamSynchronize(stream));
        for (size_t i = 0; i < cells; ++i) {
            const size_t col = i % ld;
            if (i >= (size_t)rows * ld || col < first || col >= first + owned) expected[i] = guard_bits;
        }
        if (expected != actual) {
            std::fprintf(stderr, "Q8 GSU3 mismatch T=%d N=%d K=%d ld=%d shard=%d+%d\n",
                         rows, n, k, ld, first, owned);
            throw std::runtime_error("Q8 GSU3 rounding, sum or padding");
        }
        ++checks;
    }
    return checks;
}

static void check_head_shard(int t, int full, int rows, int first, hipStream_t stream) {
    constexpr int k=2560, guard_rows=16;
    const int slice=std::min(full,(16*1024*1024/k/128)*128);
    std::mt19937 rng(971+t+full+first);
    std::uniform_real_distribution<float> random(-1.f,1.f);
    std::vector<__half> x((size_t)t*k),w((size_t)rows*k);
    for(auto& v:x)v=__float2half(random(rng));
    for(auto& v:w)v=__float2half(random(rng));
    Buffer dx(x.size()*2),dw(w.size()*2),stage((size_t)slice*k*2);
    Buffer wide((size_t)t*full*4),small(((size_t)t*rows+2*guard_rows)*4);
    ck(hipMemcpyAsync(dx.p,x.data(),x.size()*2,hipMemcpyHostToDevice,stream));
    ck(hipMemcpyAsync(dw.p,w.data(),w.size()*2,hipMemcpyHostToDevice,stream));
    ck(hipMemsetAsync(small.p,0xA5,((size_t)t*rows+2*guard_rows)*4,stream));
    for(int start=0;start<full;start+=slice) {
        const int count=std::min(slice,full-start),lo=std::max(first,start),hi=std::min(first+rows,start+count);
        if(hi<=lo)continue;
        ck(hipMemsetAsync(stage.p,0,(size_t)count*k*2,stream));
        ck(hipMemcpyAsync((__half*)stage.p+(size_t)(lo-start)*k,(__half*)dw.p+(size_t)(lo-first)*k,
            (size_t)(hi-lo)*k*2,hipMemcpyDeviceToDevice,stream));
        if(!native((uint16_t*)dx.p,(uint16_t*)stage.p,(float*)wide.p+start,{t,count,k,full-count},stream))
            throw std::runtime_error("original head slice dispatch");
    }
    if(!native((uint16_t*)dx.p,(uint16_t*)dw.p,(float*)small.p+guard_rows,{t,rows,k,0},stream))
        throw std::runtime_error("compact head dispatch");
    std::vector<float> reference((size_t)t*rows),actual((size_t)t*rows+2*guard_rows);
    ck(hipMemcpy2DAsync(reference.data(),rows*4,(float*)wide.p+first,full*4,rows*4,t,hipMemcpyDeviceToHost,stream));
    ck(hipMemcpyAsync(actual.data(),small.p,actual.size()*4,hipMemcpyDeviceToHost,stream));
    ck(hipStreamSynchronize(stream));
    if(std::memcmp(reference.data(),actual.data()+guard_rows,reference.size()*4))
        throw std::runtime_error("compact head projection changed FP32 product bits");
    for(int i=0;i<guard_rows;++i) {
        unsigned a,b;std::memcpy(&a,&actual[i],4);std::memcpy(&b,&actual[actual.size()-guard_rows+i],4);
        if(a!=0xA5A5A5A5u||b!=0xA5A5A5A5u)throw std::runtime_error("compact head guards");
    }
}

int main() {
    try {
        int cards=0;ck(hipGetDeviceCount(&cards));
        if(cards<2) throw std::runtime_error("two devices required");
        int cases=0;
        for(int dev=0;dev<2;++dev) {
            ck(hipSetDevice(dev));
            if(!strata_pf_padded_gemm_available()) throw std::runtime_error("gfx11 required");
            hipStream_t stream;ck(hipStreamCreate(&stream));hipblasHandle_t blas;cb(hipblasCreate(&blas));cb(hipblasSetStream(blas,stream));
            if (const char* exact = std::getenv("FNX_PF_HC_DOWN_GSU3"); exact && !std::strcmp(exact, "1"))
                cases += check_gsu3(320, 10240, stream, blas);
            if (const char* wide = std::getenv("FNX_PF_WIDE_GSU3"); wide && !std::strcmp(wide, "1")) {
                for (int n : {512, 640, 1664, 5760, 6528}) cases += check_gsu3(n, 2560, stream, blas);
                cases += check_gsu3(2560, 3072, stream, blas);
            }
            if (const char* q8 = std::getenv("FNX_PF_Q8_GSU3"); q8 && !std::strcmp(q8, "1")) {
                int q8_checks = check_q8_gsu3(320, 10240, 0, 320, stream);
                for (int n : {512, 640, 1664, 5760, 6528}) q8_checks += check_q8_gsu3(n, 2560, 0, n, stream);
                q8_checks += check_q8_gsu3(2560, 3072, 0, 2560, stream);
                q8_checks += check_q8_gsu3(512, 2560, 256, 256, stream);
                q8_checks += check_q8_gsu3(6528, 2560, 6144, 384, stream);
                std::printf("device=%d Q8 GSU3 exact checks=%d\n", dev, q8_checks);
            }
            const Shape shapes[]={{64,512,32,0},{65,513,96,13},{129,1031,68*32,7},
                {256,2560,320,0},{1024,640,2560,0},{4096,2560,6144,0},{1100,10240,320,0},{64,512,10240,0},
                {65,319,64,5},{65,320,64,7},{64,320,10240,0},{113,320,10240,9},{353,320,10240,0},
                {512,320,10240,7},{513,320,10240,0},{2048,320,10240,0},{4096,320,10240,0}};
            for(auto s:shapes) for(int mode:{0,1,3,4}) {check_case(s,mode,stream,blas);++cases;}
            for(int t:{64,65,1024,4096,8192})for(int card=0;card<2;++card) {
                check_head_shard(t,12288,6144,card*6144,stream);++cases;
                check_head_shard(t,512,256,card*256,stream);++cases;
            }
            check_case({64,512,32,9},2,stream,blas);++cases;
            Buffer x(64*512*2),w(512*512*2),y(64*512*4);
            ck(hipMemset(y.p,0x35,64*512*4));
            const Shape rejected[]={{63,512,32,0},{64,0,32,0},{64,512,48,0},{64,512,32,-1}};
            for(auto s:rejected) {
                if(native((uint16_t*)x.p,(uint16_t*)w.p,(float*)y.p,s,stream))
                    throw std::runtime_error("unsupported shape accepted");
                ++cases;
            }
            std::vector<unsigned char> bytes(64*512*4);ck(hipMemcpy(bytes.data(),y.p,bytes.size(),hipMemcpyDeviceToHost));
            if(!std::all_of(bytes.begin(),bytes.end(),[](unsigned char c){return c==0x35;}))
                throw std::runtime_error("fallback path wrote output");
            cb(hipblasDestroy(blas));ck(hipStreamDestroy(stream));
        }
        std::printf("prefill GEMM: %d two-card FP64 oracle, FP32 range, padding and fallback cases passed\n",cases);
    } catch(const std::exception& e) {std::fprintf(stderr,"prefill GEMM: %s\n",e.what());return 1;}
}
