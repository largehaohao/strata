#include "mtp_head_q4.hpp"
#include <fstream>
#include <iostream>
#include <numeric>

namespace {
void need(bool ok, const char* why) { if (!ok) throw std::runtime_error(why); }
template<class F> void rejects(F f) {
    try { f(); } catch (const std::runtime_error&) { return; }
    throw std::runtime_error("invalid input was accepted");
}
fnx::Tensor tensor(const std::vector<uint8_t>& bytes, int cols, int rows) {
    fnx::Tensor t; t.type=fnx::T_Q8_0; t.ne[0]=cols; t.ne[1]=rows; t.n_dims=2;
    t.data=bytes.data(); t.nbytes=bytes.size(); return t;
}
}
int main(int argc, char** argv) {
    try {
        std::vector<uint8_t> bytes(3*2*34,0);
        auto t=tensor(bytes,64,3);
        auto q=fnx::mtp_head_q4_rows(t,{2,0,2});
        need(q.size()==3*2*18,"wrong packed size");
        for (size_t b=0;b<6;++b) {
            need(q[b*18]==0 && q[b*18+1]==0x80,"zero scale");
            for(int j=2;j<18;++j) need(q[b*18+j]==0x88,"zero nibble");
        }
        need(fnx::mtp_head_q4_rows(t,{}).empty(),"empty row list");
        rejects([&]{fnx::mtp_head_q4_rows(t,{-1});});
        rejects([&]{fnx::mtp_head_q4_rows(t,{3});});
        auto bad=t;bad.type=fnx::T_BF16;rejects([&]{fnx::mtp_head_q4_rows(bad,{0});});
        bad=t;bad.data=nullptr;rejects([&]{fnx::mtp_head_q4_rows(bad,{0});});
        bad=t;--bad.nbytes;rejects([&]{fnx::mtp_head_q4_rows(bad,{0});});
        bad=t;bad.ne[0]=63;rejects([&]{fnx::mtp_head_q4_rows(bad,{0});});
        bytes[1]=0x7c;rejects([&]{fnx::mtp_head_q4_rows(t,{0});});
        bytes[1]=0x7e;rejects([&]{fnx::mtp_head_q4_rows(t,{0});});
        // Equal-magnitude maxima of opposite sign must retain the first sign.
        float x[32]{}; uint8_t a[18],b[18];x[0]=8;x[1]=-8;fnx::mtp_head_q4_block(x,a);
        std::swap(x[0],x[1]);fnx::mtp_head_q4_block(x,b);
        need((a[1]&0x80) && !(b[1]&0x80),"signed maximum tie");
        if(argc==3) {
            std::ifstream in(argv[1],std::ios::binary);
            need(in.good(),"open Q8 input");
            std::vector<uint8_t> raw((std::istreambuf_iterator<char>(in)),{});
            need(!raw.empty() && raw.size()%34==0 && raw.size()/34<=INT32_MAX,"input whole blocks");
            auto sample=tensor(raw,32,(int)(raw.size()/34));
            std::vector<int> rows(raw.size()/34);std::iota(rows.begin(),rows.end(),0);
            const auto result=fnx::mtp_head_q4_rows(sample,rows);
            std::ofstream out(argv[2],std::ios::binary);out.write((const char*)result.data(),result.size());
            need(out.good(),"write Q4 output");
            // Row gathering must preserve both order and repetitions.
            const auto gathered=fnx::mtp_head_q4_rows(sample,{(int)rows.size()-1,0,(int)rows.size()-1});
            need(std::equal(gathered.begin(),gathered.begin()+18,result.end()-18),"last row");
            need(std::equal(gathered.begin()+18,gathered.begin()+36,result.begin()),"first row");
            need(std::equal(gathered.begin()+36,gathered.end(),result.end()-18),"repeated row");
            std::cout<<"packed "<<rows.size()<<" blocks for independent reference comparison\n";
        } else need(argc==1,"usage: mtp_head_test [input-q8 output-q4]");
        std::cout<<"PASS draft-head quantization boundaries and row gathering\n";
    } catch(const std::exception& e) { std::cerr<<e.what()<<'\n';return 1; }
}
