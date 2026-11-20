// Legacy compatibility and context-only proposal contracts; no engine or GPU.
#include "lookup.hpp"
#include <iostream>
#include <random>
#include <string>

namespace {
constexpr int image_token = 1 << 30;
void require(bool ok, const char* reason) {if (!ok) throw std::runtime_error(reason);}

// The original nearest-occurrence rule is the compatibility oracle.
std::vector<int> legacy(const std::vector<int>& ctx, int lk, int d1) {
    std::vector<int> r;
    if (!lk || ctx.size() <= 3) return r;
    const size_t n0 = ctx.size() - 3;
    for (size_t s = n0; s-- > 0;)
        if (std::equal(ctx.begin()+s, ctx.begin()+s+3, ctx.begin()+n0)) {
            if (d1 < 0 || ctx[s+3] == d1) {
                int m = 3;
                while (m < lk && (size_t)(m-3) < s && ctx[s-1-(m-3)] == ctx[n0-1-(m-3)]) ++m;
                const int k = std::min(lk, std::max(3,m));
                r.assign(ctx.begin()+s+3, ctx.begin()+std::min(ctx.size(),s+3+k));
            }
            break;
        }
    r.erase(std::find_if(r.begin(),r.end(),[](int t){return t>=image_token;}),r.end());
    return r;
}

void proposal(const std::vector<int>& ctx, int lk, int d1, const std::vector<int>& r) {
    require((int)r.size() <= lk, "proposal past lookup limit");
    require(std::all_of(r.begin(),r.end(),[](int t){return t<image_token;}), "image cell proposed");
    if (r.empty()) return;
    require(d1 < 0 || r[0] == d1, "proposal does not start with sampled d1");
    require(ctx.size() > 3, "proposal without a context trigram");
    const size_t n0 = ctx.size()-3;
    bool found = false;
    for (size_t s = 0; s < n0; ++s) {
        if (s+3+r.size() > ctx.size()) continue;
        if (std::equal(ctx.begin()+s,ctx.begin()+s+3,ctx.begin()+n0) &&
            std::equal(r.begin(),r.end(),ctx.begin()+s+3)) {found=true;break;}
    }
    require(found, "proposal is not an earlier context copy");
}
}

int main() {
    try {
        const std::vector<int> mismatch = {8,9,10,1,2,3,4,5,6,7,99,50,51,1,2,3,77,78,79,88,8,9,10,1,2,3};
        require(legacy(mismatch,15,4).empty(), "nearest mismatch fixture");
        const auto recovered=fnx::lookup_drafts(mismatch,15,4,2,image_token);
        require(recovered==std::vector<int>({4,5,6,7,99,50}), "older compatible occurrence not found");
        auto longer=mismatch;longer[16]=4;longer[17]=92;longer[18]=93;
        require(fnx::lookup_drafts(longer,15,4,2,image_token).size()>legacy(longer,15,4).size(), "longer match not preferred");
        const std::vector<int> image={1,2,3,4,image_token,5,6,1,2,3};
        require(fnx::lookup_drafts(image,15,4,32,image_token)==std::vector<int>({4}), "image boundary not preserved");
        std::mt19937 rng(20261109);
        size_t checks=0;
        for (int rep=0;rep<6000;++rep) {
            const int vocab=2+rng()%40, n=rng()%400;
            std::vector<int> ctx(n);
            for (int& token:ctx) token=rng()%vocab;
            if (n>20 && rep%3) {
                const int length=3+rng()%16, at=rng()%(n-length);
                const std::vector<int> copy(ctx.begin()+at,ctx.begin()+at+length);
                std::copy(copy.begin(),copy.end(),ctx.end()-length);
            }
            if (n && rep%7==0) ctx[rng()%n]=image_token+rng()%4;
            const int lk=rng()%16, d1=rep%3 ? (int)(rng()%vocab) : -1;
            const auto old=legacy(ctx,lk,d1);
            require(fnx::lookup_drafts(ctx,lk,d1,0,image_token)==old,"default path changed");
            require(fnx::lookup_drafts(ctx,lk,d1,1,image_token)==old,"one-match path changed");
            checks+=2;
            auto previous=old;
            for (int budget:{2,8,32,64}) {
                const auto r=fnx::lookup_drafts(ctx,lk,d1,budget,image_token);
                proposal(ctx,lk,d1,r);
                require(r.size()>=previous.size(),"wider search loses a usable match");
                if (r.size()==previous.size()) require(r==previous,"recent tie changed");
                previous=r;++checks;
            }
        }
        for (int budget : {0, 1, 8, 32, 64}) {
            fnx::LookupIndex index;
            std::vector<int> history;
            for (int i = 0; i < 4096; ++i) {
                history.push_back(i % 37 == 0 ? image_token : (int)(rng() % 6));
                if (i % 3) continue;   // grow by more than one token between lookups
                for (int d1 : {-1, 3}) {
                    const auto expected = fnx::lookup_drafts(history, 15, d1, budget, image_token);
                    require(fnx::lookup_drafts(history, 15, d1, budget, image_token, &index) == expected,
                            "indexed proposal differs after growth or resize");
                    ++checks;
                }
            }
        }
        for (int bad:{-1,65}) {
            bool threw=false;
            try {(void)fnx::lookup_drafts(mismatch,15,4,bad,image_token);}
            catch (const std::runtime_error&) {threw=true;}
            require(threw,"invalid search budget accepted");++checks;
        }
        std::cout << "PASS " << checks << " lookup compatibility/proposal checks\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';return 1;
    }
}
