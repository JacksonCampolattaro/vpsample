#pragma once
#include <torch/extension.h>
#include <random>
#include <limits>

namespace random_voxel_subsample {
    struct Cell {
        uint64_t key;
        uint32_t idx;
        uint32_t weight;
        Cell* next;
    };

    struct SlowHash {
        Cell** table;
        Cell* storage;
        uint64_t size;
        uint64_t multiplier;
        uint64_t count;
        uint64_t capacity;

        SlowHash(Cell** table, Cell* storage, const uint64_t size, const uint64_t capacity)
            : table(table), storage(storage), size(size), multiplier(~0ull / size + 1), count(0),
              capacity(capacity) {}

        void insert(const uint64_t key, const uint32_t idx, const uint32_t weight) {
            const uint64_t slow_idx = __uint128_t(key * multiplier) * size >> 64;
            Cell** cell = &table[slow_idx];
            for (; *cell && (*cell)->key != key; cell = &(*cell)->next);
            if (*cell && (*cell)->weight >= weight)
                return;
            if (*cell == nullptr) {
                TORCH_CHECK(count < capacity, "voxel subsample storage exhausted");
                *cell = storage + count;
                ++count;
                (*cell)->next = nullptr;
            }
            (*cell)->idx = idx;
            (*cell)->key = key;
            (*cell)->weight = weight;
        }

        torch::Tensor finalize() {
            auto indices = torch::empty(count, torch::kInt64);
            uint64_t* pindices = (uint64_t*)indices.data_ptr();
            for (uint64_t i = 0; i < count; ++i)
                pindices[i] = storage[i].idx;
            return indices;
        }
    };

    template <uint64_t hash_size>
    struct FastHash {
        uint32_t indices[hash_size];
        uint64_t key[hash_size];
        uint32_t weight[hash_size];
        uint32_t running_weight[hash_size];
        SlowHash* slow_hash;
        std::mt19937 rand;

        FastHash(SlowHash& slow_hash)
            : slow_hash(&slow_hash) {
          rand.seed(torch::randint(0, std::numeric_limits<int32_t>::max(), 1,
                                   torch::kU64)
                        .item<int64_t>());
          for (uint64_t i = 0; i < hash_size; ++i)
            key[i] = ~0ull;
        }

        void insert(const uint64_t key_, const uint32_t idx_) {
            const uint64_t fast_idx = key_ % hash_size;
            if (key[fast_idx] == key_) {
                uint32_t new_weight = running_weight[fast_idx] * (uint32_t)31 + (uint32_t)1857864947;
                if (new_weight > weight[fast_idx]) {
                    weight[fast_idx] = new_weight;
                    indices[fast_idx] = idx_;
                }
                running_weight[fast_idx] = new_weight;
            }
            else {
                if (key[fast_idx] != ~0ull)
                    slow_hash->insert(key[fast_idx], indices[fast_idx], weight[fast_idx]);
                key[fast_idx] = key_;
                indices[fast_idx] = idx_;
                weight[fast_idx] = running_weight[fast_idx] = rand();
            }
        }

        void finalize() {
            for (uint64_t i = 0; i < hash_size; ++i)
                if (key[i] != ~0ull)
                    slow_hash->insert(key[i], indices[i], weight[i]);
        }
    };

    torch::Tensor voxel_subsample(
        const torch::Tensor& pos,
        const float voxel_size,
        torch::Tensor& hash_table
    ) {
        TORCH_CHECK(pos.is_cpu(), "voxel_subsample expects a CPU tensor");
        const double grid_size = 1 / (double)voxel_size;
        const uint64_t slow_hash_size = hash_table.size(0);
        constexpr uint64_t fast_hash_size = 2048;
        // Each cell holds at least one point, so the point count bounds the number of cells.
        auto hash_storage = torch::empty(
            {(pos.size(0) * static_cast<int64_t>(sizeof(Cell)) + 7) / 8}, torch::kInt64);
        SlowHash slow_hash(
            (Cell**)hash_table.data_ptr(),
            (Cell*)hash_storage.data_ptr(),
            slow_hash_size,
            pos.size(0));
        FastHash<fast_hash_size> fast_hash(slow_hash);
        const uint64_t pc_size = pos.size(0);
        const float* pc = (const float*)pos.data_ptr();
        const uint64_t x_step = ((1ull << 42) / slow_hash_size * slow_hash_size + uint64_t(slow_hash_size * 0.292135)) /
            fast_hash_size * fast_hash_size + 1913;
        const uint64_t z_step = ((1ull << 21) / slow_hash_size * slow_hash_size + uint64_t(slow_hash_size * 0.559518)) /
            fast_hash_size * fast_hash_size + 1187;
        const auto d2u = [](double x) {
            int64_t tmp = x;
            return *(uint64_t*)&tmp;
        };
        for (uint64_t i = 0; i < pc_size; ++i) {
            const uint64_t key = (d2u(pc[i * 3] * grid_size) * x_step) + (d2u(pc[i * 3 + 1] * grid_size)) + (d2u(
                pc[i * 3 + 2] * grid_size) * z_step);
            fast_hash.insert(key, i);
        }
        fast_hash.finalize();
        return slow_hash.finalize();
    }
}
