// Benchmark-only skinny GEMM candidates; included inside namespace pfg.
#pragma once

// Skinny HC down: 96x96 outputs, four waves, 32 columns per K step. Six
// staged vector loads and 15 KiB LDS leave more resident blocks than the
// 128x128/BK64 kernel. Each output still takes WMMA steps in increasing K.
__global__ void __launch_bounds__(128) kernel_narrow96(const _Float16* __restrict__ X,
    const _Float16* __restrict__ W, float* __restrict__ Y, int M, int N, int K,
    int ldy, int accumulate, int ldx, int ldw) {
#if defined(__gfx1100__) || defined(__gfx1101__) || defined(__gfx1102__) || defined(__gfx1103__) || defined(__gfx1150__) || defined(__gfx1151__)
    constexpr int TILE=96,STEP=32,LD=40;
    __shared__ __align__(16) _Float16 sx[TILE][LD],sw[TILE][LD];
    const int tid=threadIdx.x,lane=tid&31,wave=tid>>5,l16=lane&15,hi=lane>>4;
    const int wm=wave&1,wn=wave>>1,nm=(M+95)/96,nn=(N+95)/96;
    const int group=blockIdx.x/(GM*nn),first=group*GM,size=min(GM,nm-first);
    const int m0=(first+(blockIdx.x%(GM*nn))%size)*96,n0=((blockIdx.x%(GM*nn))/size)*96;
    const int row=tid>>2,col=8*(tid&3);
#if __has_builtin(__builtin_amdgcn_make_buffer_rsrc) && __has_builtin(__builtin_amdgcn_raw_buffer_load_b128)
    const auto xr=__builtin_amdgcn_make_buffer_rsrc(const_cast<_Float16*>(X),0,(unsigned)((size_t)M*ldx*2),0x31004000u);
    const auto wr=__builtin_amdgcn_make_buffer_rsrc(const_cast<_Float16*>(W),0,(unsigned)((size_t)N*ldw*2),0x31004000u);
    const unsigned xb=(unsigned)(m0+row)*(unsigned)ldx*2u+(unsigned)col*2u;
    const unsigned wb=(unsigned)(n0+row)*(unsigned)ldw*2u+(unsigned)col*2u;
    auto rx=[&](int r,int k) {return __builtin_bit_cast(uint4,__builtin_amdgcn_raw_buffer_load_b128(xr,xb+(unsigned)r*ldx*2u,(unsigned)k*2u,0));};
    auto rw=[&](int r,int k) {return __builtin_bit_cast(uint4,__builtin_amdgcn_raw_buffer_load_b128(wr,wb+(unsigned)r*ldw*2u,(unsigned)k*2u,0));};
#else
    auto rx=[&](int r,int k) {return *reinterpret_cast<const uint4*>(X+(size_t)min(m0+row+r,M-1)*ldx+k+col);};
    auto rw=[&](int r,int k) {return *reinterpret_cast<const uint4*>(W+(size_t)min(n0+row+r,N-1)*ldw+k+col);};
#endif
    uint4 a0,a1,a2,b0,b1,b2;
    auto load=[&](int k){a0=rx(0,k);a1=rx(32,k);a2=rx(64,k);b0=rw(0,k);b1=rw(32,k);b2=rw(64,k);};
    auto store=[&] {
        *reinterpret_cast<uint4*>(&sx[row][col])=a0;*reinterpret_cast<uint4*>(&sx[row+32][col])=a1;
        *reinterpret_cast<uint4*>(&sx[row+64][col])=a2;*reinterpret_cast<uint4*>(&sw[row][col])=b0;
        *reinterpret_cast<uint4*>(&sw[row+32][col])=b1;*reinterpret_cast<uint4*>(&sw[row+64][col])=b2;
    };
    f8 acc[3][3];
#pragma unroll
    for(int i=0;i<3;++i)
#pragma unroll
        for(int j=0;j<3;++j)acc[i][j]=f8{0,0,0,0,0,0,0,0};
    load(0);store();__syncthreads();
    const int ar=48*wm+l16,br=48*wn+l16;
    for(int k=0;k<K;k+=STEP) {
        const bool more=k+STEP<K;
        if(more)load(k+STEP);
#pragma unroll
        for(int q=0;q<STEP;q+=16) {
            const h16 a[3]={frag(&sx[ar][q]),frag(&sx[ar+16][q]),frag(&sx[ar+32][q])};
            const h16 b[3]={frag(&sw[br][q]),frag(&sw[br+16][q]),frag(&sw[br+32][q])};
#pragma unroll
            for(int i=0;i<3;++i)
#pragma unroll
                for(int j=0;j<3;++j)acc[i][j]=__builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a[i],b[j],acc[i][j]);
        }
        if(more){__syncthreads();store();__syncthreads();}
    }
#pragma unroll
    for(int i=0;i<3;++i)
#pragma unroll
        for(int j=0;j<3;++j) {
            const int n=n0+48*wn+16*j+l16;
            if(n>=N)continue;
#pragma unroll
            for(int e=0;e<8;++e) {
                const int m=m0+48*wm+16*i+2*e+hi;
                if(m<M){float* y=Y+(size_t)m*ldy+n;*y=accumulate?*y+acc[i][j][e]:acc[i][j][e];}
            }
        }
#endif
}

// Cover all 320 HC output columns without the old 384-column rounded grid.
// 64 token rows, 10 waves (two along tokens, five along output columns).
__global__ void __launch_bounds__(320) kernel_hcdown320(const _Float16* __restrict__ X,
    const _Float16* __restrict__ W, float* __restrict__ Y, int M, int N, int K,
    int ldy, int accumulate, int ldx, int ldw) {
#if defined(__gfx1100__) || defined(__gfx1101__) || defined(__gfx1102__) || defined(__gfx1103__) || defined(__gfx1150__) || defined(__gfx1151__)
    constexpr int MT=64,NT=320,STEP=64,LD=72;
    __shared__ __align__(16) _Float16 sx[MT][LD],sw[NT][LD];
    const int tid=threadIdx.x,lane=tid&31,wave=tid>>5,l16=lane&15,hi=lane>>4;
    const int wm=wave&1,wn=wave>>1,nm=(M+MT-1)/MT,nn=(N+NT-1)/NT;
    const int group=blockIdx.x/(GM*nn),first=group*GM,size=min(GM,nm-first);
    const int m0=(first+(blockIdx.x%(GM*nn))%size)*MT,n0=((blockIdx.x%(GM*nn))/size)*NT;
    const int row=tid>>3,col=8*(tid&7);
#if __has_builtin(__builtin_amdgcn_make_buffer_rsrc) && __has_builtin(__builtin_amdgcn_raw_buffer_load_b128)
    const auto xr=__builtin_amdgcn_make_buffer_rsrc(const_cast<_Float16*>(X),0,(unsigned)((size_t)M*ldx*2),0x31004000u);
    const auto wr=__builtin_amdgcn_make_buffer_rsrc(const_cast<_Float16*>(W),0,(unsigned)((size_t)N*ldw*2),0x31004000u);
    const unsigned xb=(unsigned)(m0+row)*(unsigned)ldx*2u+(unsigned)col*2u;
    const unsigned wb=(unsigned)(n0+row)*(unsigned)ldw*2u+(unsigned)col*2u;
    auto rx=[&](int r,int k) {return __builtin_bit_cast(uint4,__builtin_amdgcn_raw_buffer_load_b128(xr,xb+(unsigned)r*ldx*2u,(unsigned)k*2u,0));};
    auto rw=[&](int r,int k) {return __builtin_bit_cast(uint4,__builtin_amdgcn_raw_buffer_load_b128(wr,wb+(unsigned)r*ldw*2u,(unsigned)k*2u,0));};
#else
    auto rx=[&](int r,int k) {return *reinterpret_cast<const uint4*>(X+(size_t)min(m0+row+r,M-1)*ldx+k+col);};
    auto rw=[&](int r,int k) {return *reinterpret_cast<const uint4*>(W+(size_t)min(n0+row+r,N-1)*ldw+k+col);};
#endif
    uint4 a0,a1,b0,b1,b2,b3,b4,b5,b6,b7;
    auto load=[&](int k) {
        a0=rx(0,k);if(row+40<MT)a1=rx(40,k);
        b0=rw(0,k);b1=rw(40,k);b2=rw(80,k);b3=rw(120,k);
        b4=rw(160,k);b5=rw(200,k);b6=rw(240,k);b7=rw(280,k);
    };
    auto store=[&] {
        *reinterpret_cast<uint4*>(&sx[row][col])=a0;
        if(row+40<MT)*reinterpret_cast<uint4*>(&sx[row+40][col])=a1;
        *reinterpret_cast<uint4*>(&sw[row][col])=b0;*reinterpret_cast<uint4*>(&sw[row+40][col])=b1;
        *reinterpret_cast<uint4*>(&sw[row+80][col])=b2;*reinterpret_cast<uint4*>(&sw[row+120][col])=b3;
        *reinterpret_cast<uint4*>(&sw[row+160][col])=b4;*reinterpret_cast<uint4*>(&sw[row+200][col])=b5;
        *reinterpret_cast<uint4*>(&sw[row+240][col])=b6;*reinterpret_cast<uint4*>(&sw[row+280][col])=b7;
    };
    f8 acc[2][4];
#pragma unroll
    for(int i=0;i<2;++i)
#pragma unroll
        for(int j=0;j<4;++j)acc[i][j]=f8{0,0,0,0,0,0,0,0};
    load(0);store();__syncthreads();
    const int ar=32*wm+l16,br=64*wn+l16;
    for(int k=0;k<K;k+=STEP) {
        const bool more=k+STEP<K;if(more)load(k+STEP);
#pragma unroll
        for(int q=0;q<STEP;q+=16) {
            const h16 a[2]={frag(&sx[ar][q]),frag(&sx[ar+16][q])};
            const h16 b[4]={frag(&sw[br][q]),frag(&sw[br+16][q]),frag(&sw[br+32][q]),frag(&sw[br+48][q])};
#pragma unroll
            for(int i=0;i<2;++i)
#pragma unroll
                for(int j=0;j<4;++j)acc[i][j]=__builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a[i],b[j],acc[i][j]);
        }
        if(more){__syncthreads();store();__syncthreads();}
    }
#pragma unroll
    for(int i=0;i<2;++i)
#pragma unroll
        for(int j=0;j<4;++j) {
            const int n=n0+64*wn+16*j+l16;if(n>=N)continue;
#pragma unroll
            for(int e=0;e<8;++e) {
                const int m=m0+32*wm+16*i+2*e+hi;
                if(m<M){float* y=Y+(size_t)m*ldy+n;*y=accumulate?*y+acc[i][j][e]:acc[i][j][e];}
            }
        }
#endif
}
