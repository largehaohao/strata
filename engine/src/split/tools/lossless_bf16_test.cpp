#include "lossless_bf16.hpp"
#include <iostream>
#include <stdexcept>
int main() {
    std::vector<float> all(65536);
    std::vector<uint16_t> compact;
    for (uint32_t high=0; high<65536; ++high) {
        const uint32_t bits=high<<16; std::memcpy(&all[high],&bits,4);
    }
    if (!fnx::pack_lossless_bf16(all.data(),all.size(),compact)) throw std::runtime_error("exact BF16 was rejected");
    for (uint32_t high=0; high<65536; ++high)
        if (compact[high]!=high) throw std::runtime_error("BF16 bits changed");
    for (uint32_t low=1; low<65536; ++low) {
        const uint32_t bits=0x3f800000u|low; float value; std::memcpy(&value,&bits,4);
        if (fnx::pack_lossless_bf16(&value,1,compact)||!compact.empty()) throw std::runtime_error("inexact value accepted");
    }
    float mixed[3]={1.f,1.00000011920928955078125f,-2.f};
    if (fnx::pack_lossless_bf16(mixed,3,compact)||!compact.empty()) throw std::runtime_error("partial conversion accepted");
    std::cout<<"PASS 65536 exact BF16 bit patterns, 65535 nonrepresentable values and whole-tensor fallback\n";
}
