/*******************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 ******************************************************************************/

#ifndef MATMUL_NATIVE_BRGEMM_PLANNER_NATIVE_TUNING_FILE_HPP
#define MATMUL_NATIVE_BRGEMM_PLANNER_NATIVE_TUNING_FILE_HPP

#include <array>
#include <cstdint>
#include <iterator>
#include <memory>
#include <string>
#include <string_view>

#include "lowoha_operators/matmul/matmul_native/brgemm/planner/native_planner.hpp"
#include "lowoha_operators/matmul/matmul_native/common/native_hw.hpp"

namespace zendnnl {
namespace lowoha {
namespace matmul {
namespace native {

// The names the override and the tuning file read.

inline constexpr const char *kNativeTuningFormat
        = "zendnnl-native-brgemm-tuning";
/// The tuning-file version this build reads.
inline constexpr int64_t kNativeTuningVersion = 2;

/// Match keys of override clauses and table entries: the numeric ones
/// (ranges), the named ones (sets of names), then the flags.  The n of a
/// call whose N is not a multiple of the native blocks is its padded width
/// (make_native_problem), as are the rule features computed from it; the
/// [MATMUL.NATIVE_BRGEMM] line shows it as padded_n.  group is the group
/// size of a weight of a format of group scales only (s4;
/// native_problem_t::wei_group_size), 0 for any other.
inline constexpr const char *kNativeMatchKeys[]
        = {"m", "n", "k", "threads", "pack_nr", "group", "dtype", "src", "dst",
                "epilogue", "trans_a", "dynamic", "has_bias"};
inline constexpr int kNativeNumericMatchKeys = 6;
inline constexpr int kNativeNamedMatchKeys = 4;
/// The formats the dtype match key names, in its order: the table index
/// holds the first four in two bits.  Every format of kNativeFormats, each
/// once, so that a tuning file keys a weight of any of them.
inline constexpr native_format_t kNativeDtypeFormats[]
        = {native_format_t::bf16, native_format_t::s8, native_format_t::f32,
                native_format_t::s4, native_format_t::f16};

/// Whether kNativeDtypeFormats names each format of kNativeFormats once.
constexpr bool native_dtype_key_names_every_format() {
    if (std::size(kNativeDtypeFormats) != std::size(kNativeFormats))
        return false;
    for (const native_format_desc_t &f : kNativeFormats) {
        size_t n = 0;
        for (const native_format_t d : kNativeDtypeFormats)
            n += d == f.id ? 1 : 0;
        if (n != 1) return false;
    }
    return true;
}
static_assert(native_dtype_key_names_every_format(),
        "the dtype match key names every format once");

constexpr std::array<const char *, std::size(kNativeDtypeFormats)>
native_dtype_names() {
    std::array<const char *, std::size(kNativeDtypeFormats)> names {};
    for (size_t i = 0; i < names.size(); ++i)
        names[i] = native_format(kNativeDtypeFormats[i]).name;
    return names;
}

/// The names the named match keys dtype, src, dst and epilogue take; dtype's
/// are its formats'.
inline constexpr auto kNativeDtypeNames = native_dtype_names();
inline constexpr const char *kNativeSrcNames[]
        = {"bf16", "f32", "u8", "s8", "f16"};
inline constexpr const char *kNativeDstNames[]
        = {"f32", "bf16", "f16", "u8", "s8", "s32"};
inline constexpr const char *kNativeEpilogueNames[]
        = {"store", "bias", "postops", "gated"};

/// The problem features rules compare, and their comparisons.  esz is the
/// bytes of a weight element of the weight's format (0.5 for s4, 2 for f16);
/// a_bytes counts the source's elements at the bits the format's kernels
/// read (1 for an s8 source, 2 for an f16 one), b_bytes and slab_l2 the
/// weight's at esz; group is the match key's.
inline constexpr const char *kNativeRuleFeatures[] = {"m", "n", "k", "threads",
        "pack_nr", "bf16", "dynamic", "trans_a", "has_bias", "epilogue",
        "dst_bytes", "esz", "a_bytes", "b_bytes", "col_tiles", "m_per_thread",
        "n_per_thread", "flops_per_thread", "l2", "l3", "l3_threads", "smt",
        "cores", "a_l3", "b_l3", "slab_l2", "group"};
inline constexpr const char *kNativeRuleOps[]
        = {"<", "<=", ">", ">=", "==", "!="};

/// Hardware keys, in the order the hardware block is checked, and the
/// features the isa key names.
inline constexpr const char *kNativeHardwareKeys[] = {"vendor", "family",
        "model", "stepping", "isa", "l1d_kib", "l2_kib", "l3_kib", "l3_cores",
        "smt", "l3_threads", "cores", "logical_cpus"};
inline constexpr const char *kNativeIsaNames[]
        = {"avx512f", "avx512bf16", "avx512vnni"};

/// The clauses of a ZENDNNL_MATMUL_NATIVE_CONFIG value; bad clauses dropped
/// with one warning each.  nullptr when no clause is left.
std::unique_ptr<native_config_source_t> make_native_override_source(
        std::string_view text);

struct native_tuning_sources_t {
    std::unique_ptr<native_config_source_t> table; ///< nullptr: no entries
    std::unique_ptr<native_config_source_t> rules; ///< nullptr: no rules
};

/// Reads and checks a tuning file against hw; never throws.
native_tuning_sources_t load_native_tuning_file(
        const std::string &path, const native_hw_t &hw);

} // namespace native
} // namespace matmul
} // namespace lowoha
} // namespace zendnnl

#endif // MATMUL_NATIVE_BRGEMM_PLANNER_NATIVE_TUNING_FILE_HPP
