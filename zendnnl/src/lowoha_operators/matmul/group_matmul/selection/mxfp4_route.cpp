/*******************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *******************************************************************************/

#include "lowoha_operators/matmul/group_matmul/selection/mxfp4_route.hpp"

#include "lowoha_operators/matmul/group_matmul/common/grp_matmul_constants.hpp"
#include "lowoha_operators/matmul/group_matmul/common/grp_matmul_env.hpp"
#include "lowoha_operators/matmul/group_matmul/custom_kernel/ck_enable.hpp"
#include "lowoha_operators/matmul/group_matmul/custom_kernel/dispatch.hpp"
#include "lowoha_operators/matmul/group_matmul/executors/m_tile/group_matmul_m_tile.hpp"
#include "lowoha_operators/matmul/group_matmul/executors/n_tile/n_tile_admission.hpp"
#include "lowoha_operators/matmul/group_matmul/selection/algo_select.hpp"
#include "lowoha_operators/matmul/lowoha_matmul_utils.hpp"

namespace zendnnl {
namespace lowoha {
namespace matmul {

namespace {

// The scheduler an explicit pin requests for `phase` when it is one MXFP4
// cannot run, else 0.  Same precedence as `group_matmul_algo_select`: an
// explicit phase setting outranks the global selector.
int conflicting_scheduler_pin(grp_matmul_phase phase) {
    const auto setting = get_grp_matmul_auto_phase_setting(phase);
    const int pin = setting.has_explicit_request()
            ? (setting.pins_generic_policy() ? setting.generic_effective_algo
                                             : 0)
            : get_grp_matmul_requested_algo();
    return is_grp_matmul_generic_algo(pin) && pin != kGrpMatmulMxfp4Algo ? pin
                                                                         : 0;
}

} // namespace

status_t grp_matmul_scan_mx_types(const std::vector<int> &M,
        const std::vector<matmul_params> &params, size_t num_ops,
        bool &has_mxfp4) {
    has_mxfp4 = false;
    for (size_t i = 0; i < num_ops && i < M.size() && i < params.size(); ++i) {
        if (M[i] <= 0) continue;
        if (!mx_types_in_role(params[i])) {
            log_error("group_matmul_direct: operation ", i,
                    " uses f4_e2m1 or f8_e8m0 outside their roles: f4_e2m1 is "
                    "a weight type, f8_e8m0 the weight scale of an f4_e2m1 "
                    "weight");
            return status_t::failure;
        }
        has_mxfp4 = has_mxfp4 || params[i].dtypes.wei == data_type_t::f4_e2m1;
    }
    return status_t::success;
}

bool grp_matmul_is_mxfp4(const std::vector<int> &M,
        const std::vector<matmul_params> &params, size_t num_ops) {
    for (size_t i = 0; i < num_ops && i < M.size() && i < params.size(); ++i) {
        if (M[i] > 0) return params[i].dtypes.wei == data_type_t::f4_e2m1;
    }
    return false;
}

bool grp_matmul_mxfp4_fuses_gated_store(const std::vector<int> &M,
        const std::vector<matmul_params> &params, grp_matmul_gated_act_t act,
        data_type_t act_dtype) {
    if (act == grp_matmul_gated_act_t::none) return true;
    if (act_dtype != data_type_t::bf16) return false;
    if (act != grp_matmul_gated_act_t::silu_and_mul
            && act != grp_matmul_gated_act_t::gelu_and_mul)
        return true;
    for (size_t i = 0; i < M.size() && i < params.size(); ++i) {
        if (M[i] > 0 && params[i].dtypes.bias != data_type_t::none)
            return false;
    }
    return true;
}

status_t grp_matmul_mxfp4_admit_tight_gated_store(const std::vector<int> &M,
        const std::vector<int> &N, const std::vector<int> &ldc,
        const std::vector<matmul_params> &params, size_t num_ops,
        grp_matmul_gated_act_t act) {
    if (act == grp_matmul_gated_act_t::none || params.empty())
        return status_t::success;
    bool tight = false;
    for (size_t i = 0;
            i < num_ops && i < M.size() && i < N.size() && i < ldc.size(); ++i)
        tight = tight || (M[i] > 0 && ldc[i] < N[i]);
    if (!tight
            || grp_matmul_mxfp4_fuses_gated_store(
                    M, params, act, params[0].dtypes.dst))
        return status_t::success;
    log_error(
            "MXFP4: a gated activation into a destination narrower than N "
            "needs a store the custom kernel fuses: bf16, and no bias under "
            "silu or gelu. A destination N wide runs any other activation "
            "after the kernel.");
    return status_t::failure;
}

status_t grp_matmul_mxfp4_admit_operands(const std::vector<int> &M,
        const std::vector<int> &N, const std::vector<int> &K,
        const std::vector<matmul_params> &params, size_t num_ops) {
    namespace ck = custom_kernel;
    size_t active = 0;
    int pack_nr = 0;
    for (size_t i = 0; i < num_ops && i < M.size(); ++i) {
        if (M[i] <= 0) continue;
        ++active;
        if (i >= N.size() || i >= K.size() || i >= params.size()) {
            log_error("MXFP4: operation ", i, " has no N, K or params entry");
            return status_t::failure;
        }
        if (params[i].dtypes.wei != data_type_t::f4_e2m1) {
            log_error("MXFP4: operation ", i,
                    " is not f4_e2m1; MXFP4 experts cannot share a call with "
                    "other weight types");
            return status_t::failure;
        }
        const uint8_t *scales = nullptr;
        int scale_ld = 0;
        if (!mxfp4_scale_view(params[i], K[i], N[i], scales, scale_ld)) {
            log_error("MXFP4: operation ", i,
                    " needs f8_e8m0 weight scales {K/", kMxfp4BlockSize,
                    ", N} (K=", K[i], ", N=", N[i], ")");
            return status_t::failure;
        }
        // One pack width serves the call, chosen from its first active
        // expert, as `prepare_for_call` does.
        if (pack_nr == 0) pack_nr = ck::plan_pack_nr(K[i], N[i]);
        if ((pack_nr != ck::kNRMin && pack_nr != ck::kNRMax)
                || N[i] % pack_nr != 0) {
            log_error("MXFP4: operation ", i, " has N=", N[i],
                    ", which the custom kernel's pack width (", ck::kNRMin,
                    " or ", ck::kNRMax, ") does not divide");
            return status_t::failure;
        }
    }
    if (active > static_cast<size_t>(ck::CallContext::kMaxExperts)) {
        log_error("MXFP4: ", active, " active experts exceed the ",
                ck::CallContext::kMaxExperts,
                " the custom kernel takes in one call");
        return status_t::failure;
    }
    return status_t::success;
}

status_t grp_matmul_mxfp4_admit(const std::vector<char> &layout,
        const std::vector<bool> &transA, const std::vector<int> &M,
        const std::vector<int> &N, const std::vector<int> &K,
        const std::vector<float> &alpha, const std::vector<float> &beta,
        const std::vector<bool> &is_weights_const,
        const std::vector<matmul_params> &params, size_t num_ops,
        grp_matmul_gated_act_t fused_act, bool sequential_chain) {
    if (sequential_chain) {
        log_error("MXFP4 is supported in parallel mode only");
        return status_t::failure;
    }
    const int pin
            = conflicting_scheduler_pin(classify_grp_matmul_phase(M, num_ops));
    if (pin != 0) {
        log_error("MXFP4: the grouped scheduler is pinned to ALGO ", pin,
                ", which cannot serve MXFP4; only ALGO ", kGrpMatmulMxfp4Algo,
                " runs the custom kernel.  Unset ZENDNNL_GRP_MATMUL_ALGO and "
                "ZENDNNL_GRP_MATMUL_AUTO_{PROMPT,DECODE}_ALGO, or set them "
                "to ",
                kGrpMatmulMxfp4Algo);
        return status_t::failure;
    }
    if (!get_grp_matmul_custom_kernel()) {
        log_error(
                "MXFP4: ZENDNNL_GRP_MATMUL_CUSTOM_KERNEL=0 turns off the "
                "custom kernel, the only backend for MXFP4");
        return status_t::failure;
    }
    if (!custom_kernel::avx512vnni_available()) {
        log_error("MXFP4: the custom kernel needs AVX-512 VNNI (VPDPBUSD)");
        return status_t::isa_unsupported;
    }
    const status_t operands
            = grp_matmul_mxfp4_admit_operands(M, N, K, params, num_ops);
    if (operands != status_t::success) return operands;
    for (size_t i = 0; i < num_ops && i < M.size(); ++i) {
        if (M[i] <= 0) continue;
        if (transA[i] || alpha[i] != 1.0f || beta[i] != 0.0f) {
            log_error("MXFP4: operation ", i,
                    " needs transA=false, alpha=1 and beta=0");
            return status_t::failure;
        }
        // The whole contract, whatever the diagnostic gate: what it excludes
        // would otherwise reach the source quantization or be dropped.
        const status_t contract = validate_mxfp4_inputs(params[i], M[i], N[i],
                K[i], 1, 1, i < is_weights_const.size() && is_weights_const[i]);
        if (contract != status_t::success) return contract;
    }
    const int n = static_cast<int>(num_ops);
    if (!m_tile_supported(layout, M, params, n, /*allow_prepacked_b=*/true)
            || !detail::n_tile_supported(M, N, params, n, fused_act)) {
        log_error(
                "MXFP4: the N-tile executor cannot slice this call (layout, "
                "post-ops or quantization metadata), and only it runs the "
                "custom kernel");
        return status_t::failure;
    }
    return status_t::success;
}

} // namespace matmul
} // namespace lowoha
} // namespace zendnnl
