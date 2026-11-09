// Opt-in, two-card long-prefill query partition and exact list exchange.
#pragma once
#include "kernels.hpp"

namespace fnx {
void qsa_query_split_prepare(int card, bool peer_on, int lanes, unsigned* errors);
bool qsa_query_split(const float* q, const uint16_t* pooled, const KvPages& pages,
                     float* scores, int* cells, int* counts, int max_pools,
                     const int* pos, int rows, int card, void* stream,
                     int stride, int host_position);
}
