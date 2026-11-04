#ifdef NDEBUG
#undef NDEBUG
#endif
#include "../src/prefill_scratch_plan.hpp"
#include <cassert>
#include <cstdint>
#include <iostream>
#include <limits>

int main() {
    size_t checked=0;
    for (int cap=2;cap<=24;++cap) for (int base:{0,1})
        for (int used=0;used<=cap;++used) for (int need=used;need<=cap+1;++need)
            for (size_t bytes:{size_t(1),size_t(255),size_t(256),size_t(257),size_t(1024),size_t(4097)}) {
                const std::array<size_t,2> page_bytes{256,512};
                const auto plan=fnx::plan_prefill_scratch(base,cap,need,used,bytes,page_bytes);
                if (!plan) {++checked;continue;}
                assert(plan.begin>=base && plan.end==cap);
                for (size_t page:page_bytes) assert((size_t)plan.pages()*page>=bytes);
                std::vector<int> free;
                for(int pg=cap;pg-- >used;)free.push_back(pg);
                const auto before=free;
                assert(fnx::reserve_prefill_pages(plan,free));
                // Simulate normal KV growth through the same low-page policy.
                std::vector<int> owned;
                for(int pg=0;pg<used;++pg)owned.push_back(pg);
                while((int)owned.size()<need) {
                    assert(!free.empty());owned.push_back(free.back());free.pop_back();
                }
                for(int pg:owned)assert(pg<plan.begin);
                // A real byte write into either card's scratch cannot affect a
                // live KV page or a guard beyond the allocation.
                for(size_t page:page_bytes) {
                    std::vector<unsigned char> memory((size_t)cap*page+32,0xA7);
                    std::fill(memory.begin()+(size_t)plan.begin*page,
                              memory.begin()+(size_t)plan.begin*page+bytes,0x3C);
                    for(int pg:owned)for(size_t i=0;i<page;++i)assert(memory[(size_t)pg*page+i]==0xA7);
                    for(size_t i=(size_t)cap*page;i<memory.size();++i)assert(memory[i]==0xA7);
                }
                fnx::return_prefill_pages(plan,free);
                const auto returned=free;
                bool rejected=false;
                try {fnx::return_prefill_pages(plan,free);}catch(const std::logic_error&){rejected=true;}
                assert(rejected && free==returned);
                // Every page can subsequently be used exactly once, including
                // the tail that held scratch, up to full-context capacity.
                std::vector<int> all=owned;all.insert(all.end(),free.begin(),free.end());
                std::sort(all.begin(),all.end());assert((int)all.size()==cap);
                for(int pg=0;pg<cap;++pg)assert(all[(size_t)pg]==pg);
                auto missing=before;
                missing.erase(std::find(missing.begin(),missing.end(),plan.begin));
                const auto unchanged=missing;
                assert(!fnx::reserve_prefill_pages(plan,missing) && missing==unchanged);
                ++checked;
            }
    const std::array<size_t,2> pages{1u<<20,2u<<20};
    assert(fnx::plan_prefill_scratch(16,1047,167,167,800u<<20,pages));
    assert(!fnx::plan_prefill_scratch(16,1047,1047,1047,800u<<20,pages));
    assert(!fnx::plan_prefill_scratch(16,1047,32,32,std::numeric_limits<size_t>::max(),pages));
    assert(!fnx::plan_prefill_scratch(16,1047,32,32,100,std::array<size_t,2>{256,0}));
    assert(!fnx::plan_prefill_scratch(16,1047,32,32,100,std::array<size_t,2>{256,257}));
    auto duplicate=std::vector<int>{3,3,2,1,0};const auto before=duplicate;
    assert(!fnx::reserve_prefill_pages({2,4},duplicate) && duplicate==before);
    std::cout<<"PASS "<<checked+6<<" scratch ownership, rollback, bounds and full-capacity checks\n";
}
