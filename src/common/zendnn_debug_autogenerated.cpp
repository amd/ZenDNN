/*******************************************************************************
* Modifications Copyright (c) 2023-2025 Advanced Micro Devices, Inc. All rights reserved.
* Notified per clause 4(b) of the license.
*******************************************************************************/

/*******************************************************************************
* Copyright 2018-2022 Intel Corporation
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

// DO NOT EDIT, AUTO-GENERATED
// Use this script to update the file: scripts/generate_zendnn_debug.py

// clang-format off

#include <assert.h>

#include "zendnn_debug.h"
#include "zendnn_types.h"

const char *zendnn_status2str(zendnn_status_t v) {
    if (v == zendnn_success) {
        return "success";
    }
    if (v == zendnn_out_of_memory) {
        return "out_of_memory";
    }
    if (v == zendnn_invalid_arguments) {
        return "invalid_arguments";
    }
    if (v == zendnn_unimplemented) {
        return "unimplemented";
    }
    if (v == zendnn_iterator_ends) {
        return "iterator_ends";
    }
    if (v == zendnn_runtime_error) {
        return "runtime_error";
    }
    if (v == zendnn_not_required) {
        return "not_required";
    }
    assert(!"unknown status");
    return "unknown status";
}

const char *zendnn_dt2str(zendnn_data_type_t v) {
    if (v == zendnn_data_type_undef) {
        return "undef";
    }
    if (v == zendnn_f16) {
        return "f16";
    }
    if (v == zendnn_bf16) {
        return "bf16";
    }
    if (v == zendnn_f32) {
        return "f32";
    }
    if (v == zendnn_s32) {
        return "s32";
    }
    if (v == zendnn_s16) {
        return "s16";
    }
    if (v == zendnn_s8) {
        return "s8";
    }
    if (v == zendnn_u8) {
        return "u8";
    }
    if (v == zendnn_s4) {
        return "s4";
    }
    if (v == zendnn_u4) {
        return "u4";
    }
    assert(!"unknown dt");
    return "unknown dt";
}

const char *zendnn_fmt_kind2str(zendnn_format_kind_t v) {
    if (v == zendnn_format_kind_undef) {
        return "undef";
    }
    if (v == zendnn_format_kind_any) {
        return "any";
    }
    if (v == zendnn_blocked) {
        return "blocked";
    }
    if (v == zendnn_format_kind_wino) {
        return "wino";
    }
    if (v == zendnn_format_kind_rnn_packed) {
        return "rnn_packed";
    }
    assert(!"unknown fmt_kind");
    return "unknown fmt_kind";
}

const char *zendnn_fmt_tag2str(zendnn_format_tag_t v) {
    if (v == zendnn_format_tag_undef) {
        return "undef";
    }
    if (v == zendnn_format_tag_any) {
        return "any";
    }
    if (v == zendnn_a) {
        return "a";
    }
    if (v == zendnn_ab) {
        return "ab";
    }
    if (v == zendnn_abc) {
        return "abc";
    }
    if (v == zendnn_abcd) {
        return "abcd";
    }
    if (v == zendnn_acbd) {
        return "acbd";
    }
    if (v == zendnn_abcde) {
        return "abcde";
    }
    if (v == zendnn_abcdef) {
        return "abcdef";
    }
    if (v == zendnn_abcdefg) {
        return "abcdefg";
    }
    if (v == zendnn_abcdefgh) {
        return "abcdefgh";
    }
    if (v == zendnn_abcdefghi) {
        return "abcdefghi";
    }
    if (v == zendnn_abcdefghij) {
        return "abcdefghij";
    }
    if (v == zendnn_abcdefghijk) {
        return "abcdefghijk";
    }
    if (v == zendnn_abcdefghijkl) {
        return "abcdefghijkl";
    }
    if (v == zendnn_abdc) {
        return "abdc";
    }
    if (v == zendnn_abdec) {
        return "abdec";
    }
    if (v == zendnn_acb) {
        return "acb";
    }
    if (v == zendnn_acbde) {
        return "acbde";
    }
    if (v == zendnn_acbdef) {
        return "acbdef";
    }
    if (v == zendnn_acdb) {
        return "acdb";
    }
    if (v == zendnn_acdeb) {
        return "acdeb";
    }
    if (v == zendnn_ba) {
        return "ba";
    }
    if (v == zendnn_bac) {
        return "bac";
    }
    if (v == zendnn_bacd) {
        return "bacd";
    }
    if (v == zendnn_bacde) {
        return "bacde";
    }
    if (v == zendnn_bca) {
        return "bca";
    }
    if (v == zendnn_bcda) {
        return "bcda";
    }
    if (v == zendnn_bcdea) {
        return "bcdea";
    }
    if (v == zendnn_cba) {
        return "cba";
    }
    if (v == zendnn_cdba) {
        return "cdba";
    }
    if (v == zendnn_dcab) {
        return "dcab";
    }
    if (v == zendnn_cdeba) {
        return "cdeba";
    }
    if (v == zendnn_decab) {
        return "decab";
    }
    if (v == zendnn_defcab) {
        return "defcab";
    }
    if (v == zendnn_abced) {
        return "abced";
    }
    if (v == zendnn_abcdfe) {
        return "abcdfe";
    }
    if (v == zendnn_abcdegf) {
        return "abcdegf";
    }
    if (v == zendnn_abcdefhg) {
        return "abcdefhg";
    }
    if (v == zendnn_abcdefgih) {
        return "abcdefgih";
    }
    if (v == zendnn_abcdefghji) {
        return "abcdefghji";
    }
    if (v == zendnn_abcdefghikj) {
        return "abcdefghikj";
    }
    if (v == zendnn_abcdefghijlk) {
        return "abcdefghijlk";
    }
    if (v == zendnn_Abc16a) {
        return "Abc16a";
    }
    if (v == zendnn_ABc16a16b) {
        return "ABc16a16b";
    }
    if (v == zendnn_ABc32a32b) {
        return "ABc32a32b";
    }
    if (v == zendnn_ABc4a4b) {
        return "ABc4a4b";
    }
    if (v == zendnn_aBc16b) {
        return "aBc16b";
    }
    if (v == zendnn_ABc16b16a) {
        return "ABc16b16a";
    }
    if (v == zendnn_Abc4a) {
        return "Abc4a";
    }
    if (v == zendnn_aBc32b) {
        return "aBc32b";
    }
    if (v == zendnn_aBc4b) {
        return "aBc4b";
    }
    if (v == zendnn_ABc4b16a4b) {
        return "ABc4b16a4b";
    }
    if (v == zendnn_ABc2b8a4b) {
        return "ABc2b8a4b";
    }
    if (v == zendnn_ABc16b16a4b) {
        return "ABc16b16a4b";
    }
    if (v == zendnn_ABc16b16a2b) {
        return "ABc16b16a2b";
    }
    if (v == zendnn_ABc4b4a) {
        return "ABc4b4a";
    }
    if (v == zendnn_ABc8a16b2a) {
        return "ABc8a16b2a";
    }
    if (v == zendnn_ABc8a8b) {
        return "ABc8a8b";
    }
    if (v == zendnn_ABc8a4b) {
        return "ABc8a4b";
    }
    if (v == zendnn_aBc8b) {
        return "aBc8b";
    }
    if (v == zendnn_ABc8b16a2b) {
        return "ABc8b16a2b";
    }
    if (v == zendnn_BAc8a16b2a) {
        return "BAc8a16b2a";
    }
    if (v == zendnn_ABc8b8a) {
        return "ABc8b8a";
    }
    if (v == zendnn_Abcd16a) {
        return "Abcd16a";
    }
    if (v == zendnn_Abcd8a) {
        return "Abcd8a";
    }
    if (v == zendnn_ABcd16a16b) {
        return "ABcd16a16b";
    }
    if (v == zendnn_Abcd32a) {
        return "Abcd32a";
    }
    if (v == zendnn_ABcd32a32b) {
        return "ABcd32a32b";
    }
    if (v == zendnn_aBcd16b) {
        return "aBcd16b";
    }
    if (v == zendnn_ABcd16b16a) {
        return "ABcd16b16a";
    }
    if (v == zendnn_aBCd16b16c) {
        return "aBCd16b16c";
    }
    if (v == zendnn_aBCd16c16b) {
        return "aBCd16c16b";
    }
    if (v == zendnn_Abcd4a) {
        return "Abcd4a";
    }
    if (v == zendnn_aBcd32b) {
        return "aBcd32b";
    }
    if (v == zendnn_aBcd4b) {
        return "aBcd4b";
    }
    if (v == zendnn_ABcd4b16a4b) {
        return "ABcd4b16a4b";
    }
    if (v == zendnn_ABcd16b16a4b) {
        return "ABcd16b16a4b";
    }
    if (v == zendnn_ABcd16b16a2b) {
        return "ABcd16b16a2b";
    }
    if (v == zendnn_ABcd4b4a) {
        return "ABcd4b4a";
    }
    if (v == zendnn_ABcd4a4b) {
        return "ABcd4a4b";
    }
    if (v == zendnn_aBCd2c4b2c) {
        return "aBCd2c4b2c";
    }
    if (v == zendnn_aBCd4b8c2b) {
        return "aBCd4b8c2b";
    }
    if (v == zendnn_aBCd4c16b4c) {
        return "aBCd4c16b4c";
    }
    if (v == zendnn_aBCd2c8b4c) {
        return "aBCd2c8b4c";
    }
    if (v == zendnn_aBCd16c16b4c) {
        return "aBCd16c16b4c";
    }
    if (v == zendnn_aBCd16c16b2c) {
        return "aBCd16c16b2c";
    }
    if (v == zendnn_aBCd4c4b) {
        return "aBCd4c4b";
    }
    if (v == zendnn_aBCd4b4c) {
        return "aBCd4b4c";
    }
    if (v == zendnn_ABcd8a16b2a) {
        return "ABcd8a16b2a";
    }
    if (v == zendnn_ABcd2b8a4b) {
        return "ABcd2b8a4b";
    }
    if (v == zendnn_ABcd8a8b) {
        return "ABcd8a8b";
    }
    if (v == zendnn_ABcd8a4b) {
        return "ABcd8a4b";
    }
    if (v == zendnn_aBcd8b) {
        return "aBcd8b";
    }
    if (v == zendnn_aBCd4c8b2c) {
        return "aBCd4c8b2c";
    }
    if (v == zendnn_ABcd8b16a2b) {
        return "ABcd8b16a2b";
    }
    if (v == zendnn_aBCd8b16c2b) {
        return "aBCd8b16c2b";
    }
    if (v == zendnn_BAcd8a16b2a) {
        return "BAcd8a16b2a";
    }
    if (v == zendnn_ABcd8b8a) {
        return "ABcd8b8a";
    }
    if (v == zendnn_aBCd8b8c) {
        return "aBCd8b8c";
    }
    if (v == zendnn_aBCd8b4c) {
        return "aBCd8b4c";
    }
    if (v == zendnn_aBCd8c16b2c) {
        return "aBCd8c16b2c";
    }
    if (v == zendnn_ABcde8a16b2a) {
        return "ABcde8a16b2a";
    }
    if (v == zendnn_aCBd8b16c2b) {
        return "aCBd8b16c2b";
    }
    if (v == zendnn_aBCd8c8b) {
        return "aBCd8c8b";
    }
    if (v == zendnn_Abcde16a) {
        return "Abcde16a";
    }
    if (v == zendnn_Abcde32a) {
        return "Abcde32a";
    }
    if (v == zendnn_ABcde16a16b) {
        return "ABcde16a16b";
    }
    if (v == zendnn_BAcde8a16b2a) {
        return "BAcde8a16b2a";
    }
    if (v == zendnn_aBCd2b4c2b) {
        return "aBCd2b4c2b";
    }
    if (v == zendnn_ABcde4b16a4b) {
        return "ABcde4b16a4b";
    }
    if (v == zendnn_ABcde2b8a4b) {
        return "ABcde2b8a4b";
    }
    if (v == zendnn_aBcde16b) {
        return "aBcde16b";
    }
    if (v == zendnn_ABcde16b16a) {
        return "ABcde16b16a";
    }
    if (v == zendnn_aBCde16b16c) {
        return "aBCde16b16c";
    }
    if (v == zendnn_aBCde16c16b) {
        return "aBCde16c16b";
    }
    if (v == zendnn_aBCde2c8b4c) {
        return "aBCde2c8b4c";
    }
    if (v == zendnn_Abcde4a) {
        return "Abcde4a";
    }
    if (v == zendnn_aBcde32b) {
        return "aBcde32b";
    }
    if (v == zendnn_aBcde4b) {
        return "aBcde4b";
    }
    if (v == zendnn_ABcde4b4a) {
        return "ABcde4b4a";
    }
    if (v == zendnn_ABcde4a4b) {
        return "ABcde4a4b";
    }
    if (v == zendnn_aBCde4b4c) {
        return "aBCde4b4c";
    }
    if (v == zendnn_aBCde2c4b2c) {
        return "aBCde2c4b2c";
    }
    if (v == zendnn_aBCde4b8c2b) {
        return "aBCde4b8c2b";
    }
    if (v == zendnn_aBCde4c16b4c) {
        return "aBCde4c16b4c";
    }
    if (v == zendnn_aBCde16c16b4c) {
        return "aBCde16c16b4c";
    }
    if (v == zendnn_aBCde16c16b2c) {
        return "aBCde16c16b2c";
    }
    if (v == zendnn_aBCde4c4b) {
        return "aBCde4c4b";
    }
    if (v == zendnn_Abcde8a) {
        return "Abcde8a";
    }
    if (v == zendnn_ABcde8a8b) {
        return "ABcde8a8b";
    }
    if (v == zendnn_ABcde8a4b) {
        return "ABcde8a4b";
    }
    if (v == zendnn_BAcde16b16a) {
        return "BAcde16b16a";
    }
    if (v == zendnn_aBcde8b) {
        return "aBcde8b";
    }
    if (v == zendnn_ABcde8b16a2b) {
        return "ABcde8b16a2b";
    }
    if (v == zendnn_aBCde8b16c2b) {
        return "aBCde8b16c2b";
    }
    if (v == zendnn_aBCde4c8b2c) {
        return "aBCde4c8b2c";
    }
    if (v == zendnn_aCBde8b16c2b) {
        return "aCBde8b16c2b";
    }
    if (v == zendnn_ABcde8b8a) {
        return "ABcde8b8a";
    }
    if (v == zendnn_ABcde32a32b) {
        return "ABcde32a32b";
    }
    if (v == zendnn_aBCde8b8c) {
        return "aBCde8b8c";
    }
    if (v == zendnn_aBCde8b4c) {
        return "aBCde8b4c";
    }
    if (v == zendnn_ABc4a8b8a4b) {
        return "ABc4a8b8a4b";
    }
    if (v == zendnn_ABcd4a8b8a4b) {
        return "ABcd4a8b8a4b";
    }
    if (v == zendnn_ABcde4a8b8a4b) {
        return "ABcde4a8b8a4b";
    }
    if (v == zendnn_BAc4b8a8b4a) {
        return "BAc4b8a8b4a";
    }
    if (v == zendnn_BAcd4b8a8b4a) {
        return "BAcd4b8a8b4a";
    }
    if (v == zendnn_BAcde4b8a8b4a) {
        return "BAcde4b8a8b4a";
    }
    if (v == zendnn_ABcd2a8b8a2b) {
        return "ABcd2a8b8a2b";
    }
    if (v == zendnn_aBCd4b8c8b4c) {
        return "aBCd4b8c8b4c";
    }
    if (v == zendnn_aBCde4b8c8b4c) {
        return "aBCde4b8c8b4c";
    }
    if (v == zendnn_aBCde2b8c8b2c) {
        return "aBCde2b8c8b2c";
    }
    if (v == zendnn_aBCde8c16b2c) {
        return "aBCde8c16b2c";
    }
    if (v == zendnn_aBCde8c8b) {
        return "aBCde8c8b";
    }
    if (v == zendnn_aBCde2b4c2b) {
        return "aBCde2b4c2b";
    }
    if (v == zendnn_aBcdef16b) {
        return "aBcdef16b";
    }
    if (v == zendnn_aBCdef16b16c) {
        return "aBCdef16b16c";
    }
    if (v == zendnn_aBCdef16c16b) {
        return "aBCdef16c16b";
    }
    if (v == zendnn_aBCdef4c16b4c) {
        return "aBCdef4c16b4c";
    }
    if (v == zendnn_aBCdef2c8b4c) {
        return "aBCdef2c8b4c";
    }
    if (v == zendnn_aBCdef4c8b2c) {
        return "aBCdef4c8b2c";
    }
    if (v == zendnn_aBCdef2b4c2b) {
        return "aBCdef2b4c2b";
    }
    if (v == zendnn_aBcdef4b) {
        return "aBcdef4b";
    }
    if (v == zendnn_aBCdef4c4b) {
        return "aBCdef4c4b";
    }
    if (v == zendnn_aBCdef4b4c) {
        return "aBCdef4b4c";
    }
    if (v == zendnn_aBCdef2c4b2c) {
        return "aBCdef2c4b2c";
    }
    if (v == zendnn_aBCdef4b8c2b) {
        return "aBCdef4b8c2b";
    }
    if (v == zendnn_aBCdef8b8c) {
        return "aBCdef8b8c";
    }
    if (v == zendnn_aBCdef8b4c) {
        return "aBCdef8b4c";
    }
    if (v == zendnn_aBCdef8c16b2c) {
        return "aBCdef8c16b2c";
    }
    if (v == zendnn_aBCdef4b8c8b4c) {
        return "aBCdef4b8c8b4c";
    }
    if (v == zendnn_aBCdef8b16c2b) {
        return "aBCdef8b16c2b";
    }
    if (v == zendnn_aCBdef8b16c2b) {
        return "aCBdef8b16c2b";
    }
    if (v == zendnn_aBCdef8c8b) {
        return "aBCdef8c8b";
    }
    if (v == zendnn_aBdc16b) {
        return "aBdc16b";
    }
    if (v == zendnn_aBdC16b2c) {
        return "aBdC16b2c";
    }
    if (v == zendnn_aBdC16b4c) {
        return "aBdC16b4c";
    }
    if (v == zendnn_aBdc4b) {
        return "aBdc4b";
    }
    if (v == zendnn_aBdc8b) {
        return "aBdc8b";
    }
    if (v == zendnn_aBdec16b) {
        return "aBdec16b";
    }
    if (v == zendnn_aBdeC16b2c) {
        return "aBdeC16b2c";
    }
    if (v == zendnn_aBdeC16b4c) {
        return "aBdeC16b4c";
    }
    if (v == zendnn_aBdec32b) {
        return "aBdec32b";
    }
    if (v == zendnn_aBdec4b) {
        return "aBdec4b";
    }
    if (v == zendnn_aBdec8b) {
        return "aBdec8b";
    }
    if (v == zendnn_aBdefc16b) {
        return "aBdefc16b";
    }
    if (v == zendnn_aBdefC16b2c) {
        return "aBdefC16b2c";
    }
    if (v == zendnn_aCBdef16c16b) {
        return "aCBdef16c16b";
    }
    if (v == zendnn_aBdefc4b) {
        return "aBdefc4b";
    }
    if (v == zendnn_aBdefc8b) {
        return "aBdefc8b";
    }
    if (v == zendnn_Abcdef16a) {
        return "Abcdef16a";
    }
    if (v == zendnn_Abcdef32a) {
        return "Abcdef32a";
    }
    if (v == zendnn_aBedc16b) {
        return "aBedc16b";
    }
    if (v == zendnn_Acb16a) {
        return "Acb16a";
    }
    if (v == zendnn_AcB16a2b) {
        return "AcB16a2b";
    }
    if (v == zendnn_AcB16a4b) {
        return "AcB16a4b";
    }
    if (v == zendnn_Acb4a) {
        return "Acb4a";
    }
    if (v == zendnn_Acb8a) {
        return "Acb8a";
    }
    if (v == zendnn_aCBd16b16c) {
        return "aCBd16b16c";
    }
    if (v == zendnn_aCBd16c16b) {
        return "aCBd16c16b";
    }
    if (v == zendnn_aCBde16b16c) {
        return "aCBde16b16c";
    }
    if (v == zendnn_aCBde16c16b) {
        return "aCBde16c16b";
    }
    if (v == zendnn_Acdb16a) {
        return "Acdb16a";
    }
    if (v == zendnn_AcdB16a2b) {
        return "AcdB16a2b";
    }
    if (v == zendnn_AcdB16a4b) {
        return "AcdB16a4b";
    }
    if (v == zendnn_Acdb32a) {
        return "Acdb32a";
    }
    if (v == zendnn_Acdb4a) {
        return "Acdb4a";
    }
    if (v == zendnn_Acdb8a) {
        return "Acdb8a";
    }
    if (v == zendnn_Acdeb16a) {
        return "Acdeb16a";
    }
    if (v == zendnn_AcdeB16a2b) {
        return "AcdeB16a2b";
    }
    if (v == zendnn_Acdeb4a) {
        return "Acdeb4a";
    }
    if (v == zendnn_Acdeb8a) {
        return "Acdeb8a";
    }
    if (v == zendnn_Adcb16a) {
        return "Adcb16a";
    }
    if (v == zendnn_BAc16a16b) {
        return "BAc16a16b";
    }
    if (v == zendnn_BAc16b16a) {
        return "BAc16b16a";
    }
    if (v == zendnn_BAcd16a16b) {
        return "BAcd16a16b";
    }
    if (v == zendnn_BAcd16b16a) {
        return "BAcd16b16a";
    }
    if (v == zendnn_aCBd4c8b8c4b) {
        return "aCBd4c8b8c4b";
    }
    if (v == zendnn_aCBde4c8b8c4b) {
        return "aCBde4c8b8c4b";
    }
    if (v == zendnn_aCBdef4c8b8c4b) {
        return "aCBdef4c8b8c4b";
    }
    if (v == zendnn_BAcde16a16b) {
        return "BAcde16a16b";
    }
    if (v == zendnn_aCBdef16b16c) {
        return "aCBdef16b16c";
    }
    if (v == zendnn_abdfce) {
        return "abdfce";
    }
    if (v == zendnn_abdefc) {
        return "abdefc";
    }
    if (v == zendnn_ABc16b32a) {
        return "ABc16b32a";
    }
    if (v == zendnn_ABc16b64a) {
        return "ABc16b64a";
    }
    if (v == zendnn_ABc4b32a4b) {
        return "ABc4b32a4b";
    }
    if (v == zendnn_ABc4b64a4b) {
        return "ABc4b64a4b";
    }
    if (v == zendnn_ABc8b32a2b) {
        return "ABc8b32a2b";
    }
    if (v == zendnn_ABc8b64a2b) {
        return "ABc8b64a2b";
    }
    if (v == zendnn_AB16b16a) {
        return "AB16b16a";
    }
    if (v == zendnn_AB16b32a) {
        return "AB16b32a";
    }
    if (v == zendnn_AB16b64a) {
        return "AB16b64a";
    }
    if (v == zendnn_AB8b16a2b) {
        return "AB8b16a2b";
    }
    if (v == zendnn_AB8b32a2b) {
        return "AB8b32a2b";
    }
    if (v == zendnn_AB8b64a2b) {
        return "AB8b64a2b";
    }
    if (v == zendnn_AB4b16a4b) {
        return "AB4b16a4b";
    }
    if (v == zendnn_AB4b32a4b) {
        return "AB4b32a4b";
    }
    if (v == zendnn_AB4b64a4b) {
        return "AB4b64a4b";
    }
    if (v == zendnn_AB16b16a4b) {
        return "AB16b16a4b";
    }
    if (v == zendnn_ABcd16b32a) {
        return "ABcd16b32a";
    }
    if (v == zendnn_ABcd16b64a) {
        return "ABcd16b64a";
    }
    if (v == zendnn_ABcd4b32a4b) {
        return "ABcd4b32a4b";
    }
    if (v == zendnn_ABcd4b64a4b) {
        return "ABcd4b64a4b";
    }
    if (v == zendnn_ABcd8b32a2b) {
        return "ABcd8b32a2b";
    }
    if (v == zendnn_ABcd8b64a2b) {
        return "ABcd8b64a2b";
    }
    if (v == zendnn_ABcde4b32a4b) {
        return "ABcde4b32a4b";
    }
    if (v == zendnn_ABcde4b64a4b) {
        return "ABcde4b64a4b";
    }
    if (v == zendnn_ABcde16b16a4b) {
        return "ABcde16b16a4b";
    }
    if (v == zendnn_ABcde16b16a2b) {
        return "ABcde16b16a2b";
    }
    if (v == zendnn_ABcde16b32a) {
        return "ABcde16b32a";
    }
    if (v == zendnn_ABcde16b64a) {
        return "ABcde16b64a";
    }
    if (v == zendnn_ABcde8b32a2b) {
        return "ABcde8b32a2b";
    }
    if (v == zendnn_ABcde8b64a2b) {
        return "ABcde8b64a2b";
    }
    if (v == zendnn_aBCdef16c16b4c) {
        return "aBCdef16c16b4c";
    }
    if (v == zendnn_aBCdef16c16b2c) {
        return "aBCdef16c16b2c";
    }
    if (v == zendnn_AB32a32b8a4b) {
        return "AB32a32b8a4b";
    }
    if (v == zendnn_AB8a4b) {
        return "AB8a4b";
    }
    if (v == zendnn_AB32a32b8a2b) {
        return "AB32a32b8a2b";
    }
    if (v == zendnn_AB8a2b) {
        return "AB8a2b";
    }
    if (v == zendnn_abDc32d) {
        return "abDc32d";
    }
    if (v == zendnn_abDC32d4c) {
        return "abDC32d4c";
    }
    if (v == zendnn_abdEc32e) {
        return "abdEc32e";
    }
    if (v == zendnn_abdEC32e2c) {
        return "abdEC32e2c";
    }
    if (v == zendnn_abdEC32e4c) {
        return "abdEC32e4c";
    }
    if (v == zendnn_aBdefC16b4c) {
        return "aBdefC16b4c";
    }
    if (v == zendnn_AcdeB16a4b) {
        return "AcdeB16a4b";
    }
    if (v == zendnn_ABcd16a16b2a) {
        return "ABcd16a16b2a";
    }
    if (v == zendnn_ABc16a16b2a) {
        return "ABc16a16b2a";
    }
    if (v == zendnn_aBCd16b16c2b) {
        return "aBCd16b16c2b";
    }
    if (v == zendnn_aBCde16b16c2b) {
        return "aBCde16b16c2b";
    }
    if (v == zendnn_Acb32a) {
        return "Acb32a";
    }
    if (v == zendnn_AcB32a2b) {
        return "AcB32a2b";
    }
    if (v == zendnn_AcB32a4b) {
        return "AcB32a4b";
    }
    if (v == zendnn_Acb48a) {
        return "Acb48a";
    }
    if (v == zendnn_AcB48a2b) {
        return "AcB48a2b";
    }
    if (v == zendnn_AcB48a4b) {
        return "AcB48a4b";
    }
    if (v == zendnn_Acb64a) {
        return "Acb64a";
    }
    if (v == zendnn_AcB64a2b) {
        return "AcB64a2b";
    }
    if (v == zendnn_AcB64a4b) {
        return "AcB64a4b";
    }
    if (v == zendnn_cBa2b) {
        return "cBa2b";
    }
    if (v == zendnn_cBa4b) {
        return "cBa4b";
    }
    if (v == zendnn_aBdc32b) {
        return "aBdc32b";
    }
    if (v == zendnn_aBdC32b2c) {
        return "aBdC32b2c";
    }
    if (v == zendnn_aBdC32b4c) {
        return "aBdC32b4c";
    }
    if (v == zendnn_aBdc48b) {
        return "aBdc48b";
    }
    if (v == zendnn_aBdC48b2c) {
        return "aBdC48b2c";
    }
    if (v == zendnn_aBdC48b4c) {
        return "aBdC48b4c";
    }
    if (v == zendnn_aBdc64b) {
        return "aBdc64b";
    }
    if (v == zendnn_aBdC64b2c) {
        return "aBdC64b2c";
    }
    if (v == zendnn_aBdC64b4c) {
        return "aBdC64b4c";
    }
    if (v == zendnn_adcb) {
        return "adcb";
    }
    if (v == zendnn_adCb2c) {
        return "adCb2c";
    }
    if (v == zendnn_adCb4c) {
        return "adCb4c";
    }
    if (v == zendnn_AcdB32a2b) {
        return "AcdB32a2b";
    }
    if (v == zendnn_AcdB32a4b) {
        return "AcdB32a4b";
    }
    if (v == zendnn_Acdb48a) {
        return "Acdb48a";
    }
    if (v == zendnn_AcdB48a2b) {
        return "AcdB48a2b";
    }
    if (v == zendnn_AcdB48a4b) {
        return "AcdB48a4b";
    }
    if (v == zendnn_Acdb64a) {
        return "Acdb64a";
    }
    if (v == zendnn_AcdB64a2b) {
        return "AcdB64a2b";
    }
    if (v == zendnn_AcdB64a4b) {
        return "AcdB64a4b";
    }
    if (v == zendnn_cdBa2b) {
        return "cdBa2b";
    }
    if (v == zendnn_cdBa4b) {
        return "cdBa4b";
    }
    if (v == zendnn_aBdeC32b2c) {
        return "aBdeC32b2c";
    }
    if (v == zendnn_aBdeC32b4c) {
        return "aBdeC32b4c";
    }
    if (v == zendnn_aBdec48b) {
        return "aBdec48b";
    }
    if (v == zendnn_aBdeC48b2c) {
        return "aBdeC48b2c";
    }
    if (v == zendnn_aBdeC48b4c) {
        return "aBdeC48b4c";
    }
    if (v == zendnn_aBdec64b) {
        return "aBdec64b";
    }
    if (v == zendnn_aBdeC64b2c) {
        return "aBdeC64b2c";
    }
    if (v == zendnn_aBdeC64b4c) {
        return "aBdeC64b4c";
    }
    if (v == zendnn_adecb) {
        return "adecb";
    }
    if (v == zendnn_adeCb2c) {
        return "adeCb2c";
    }
    if (v == zendnn_adeCb4c) {
        return "adeCb4c";
    }
    if (v == zendnn_Acdeb32a) {
        return "Acdeb32a";
    }
    if (v == zendnn_AcdeB32a2b) {
        return "AcdeB32a2b";
    }
    if (v == zendnn_AcdeB32a4b) {
        return "AcdeB32a4b";
    }
    if (v == zendnn_Acdeb48a) {
        return "Acdeb48a";
    }
    if (v == zendnn_AcdeB48a2b) {
        return "AcdeB48a2b";
    }
    if (v == zendnn_AcdeB48a4b) {
        return "AcdeB48a4b";
    }
    if (v == zendnn_Acdeb64a) {
        return "Acdeb64a";
    }
    if (v == zendnn_AcdeB64a2b) {
        return "AcdeB64a2b";
    }
    if (v == zendnn_AcdeB64a4b) {
        return "AcdeB64a4b";
    }
    if (v == zendnn_cdeBa2b) {
        return "cdeBa2b";
    }
    if (v == zendnn_cdeBa4b) {
        return "cdeBa4b";
    }
    if (v == zendnn_aBdefc32b) {
        return "aBdefc32b";
    }
    if (v == zendnn_aBdefC32b2c) {
        return "aBdefC32b2c";
    }
    if (v == zendnn_aBdefC32b4c) {
        return "aBdefC32b4c";
    }
    if (v == zendnn_aBdefc48b) {
        return "aBdefc48b";
    }
    if (v == zendnn_aBdefC48b2c) {
        return "aBdefC48b2c";
    }
    if (v == zendnn_aBdefC48b4c) {
        return "aBdefC48b4c";
    }
    if (v == zendnn_aBdefc64b) {
        return "aBdefc64b";
    }
    if (v == zendnn_aBdefC64b2c) {
        return "aBdefC64b2c";
    }
    if (v == zendnn_aBdefC64b4c) {
        return "aBdefC64b4c";
    }
    if (v == zendnn_adefcb) {
        return "adefcb";
    }
    if (v == zendnn_adefCb2c) {
        return "adefCb2c";
    }
    if (v == zendnn_adefCb4c) {
        return "adefCb4c";
    }
    if (v == zendnn_AB16b32a4b) {
        return "AB16b32a4b";
    }
    if (v == zendnn_AB16b48a4b) {
        return "AB16b48a4b";
    }
    if (v == zendnn_AB16b64a4b) {
        return "AB16b64a4b";
    }
    if (v == zendnn_AB16b16a2b) {
        return "AB16b16a2b";
    }
    if (v == zendnn_AB16b32a2b) {
        return "AB16b32a2b";
    }
    if (v == zendnn_AB16b48a2b) {
        return "AB16b48a2b";
    }
    if (v == zendnn_AB16b64a2b) {
        return "AB16b64a2b";
    }
    if (v == zendnn_ABc16b32a4b) {
        return "ABc16b32a4b";
    }
    if (v == zendnn_ABc16b48a4b) {
        return "ABc16b48a4b";
    }
    if (v == zendnn_ABc16b64a4b) {
        return "ABc16b64a4b";
    }
    if (v == zendnn_ABc16b32a2b) {
        return "ABc16b32a2b";
    }
    if (v == zendnn_ABc16b48a2b) {
        return "ABc16b48a2b";
    }
    if (v == zendnn_ABc16b64a2b) {
        return "ABc16b64a2b";
    }
    if (v == zendnn_ABcd16b32a4b) {
        return "ABcd16b32a4b";
    }
    if (v == zendnn_ABcd16b48a4b) {
        return "ABcd16b48a4b";
    }
    if (v == zendnn_ABcd16b64a4b) {
        return "ABcd16b64a4b";
    }
    if (v == zendnn_ABcd16b32a2b) {
        return "ABcd16b32a2b";
    }
    if (v == zendnn_ABcd16b48a2b) {
        return "ABcd16b48a2b";
    }
    if (v == zendnn_ABcd16b64a2b) {
        return "ABcd16b64a2b";
    }
    if (v == zendnn_ABcde16b32a4b) {
        return "ABcde16b32a4b";
    }
    if (v == zendnn_ABcde16b48a4b) {
        return "ABcde16b48a4b";
    }
    if (v == zendnn_ABcde16b64a4b) {
        return "ABcde16b64a4b";
    }
    if (v == zendnn_ABcde16b32a2b) {
        return "ABcde16b32a2b";
    }
    if (v == zendnn_ABcde16b48a2b) {
        return "ABcde16b48a2b";
    }
    if (v == zendnn_ABcde16b64a2b) {
        return "ABcde16b64a2b";
    }
    if (v == zendnn_ABc32a16b) {
        return "ABc32a16b";
    }
    if (v == zendnn_ABcd32a16b) {
        return "ABcd32a16b";
    }
    if (v == zendnn_ABcde32a16b) {
        return "ABcde32a16b";
    }
    if (v == zendnn_AB48a16b) {
        return "AB48a16b";
    }
    if (v == zendnn_AB48a32b) {
        return "AB48a32b";
    }
    if (v == zendnn_ABc40a16b) {
        return "ABc40a16b";
    }
    if (v == zendnn_ABc40a32b) {
        return "ABc40a32b";
    }
    if (v == zendnn_aBC48b16c) {
        return "aBC48b16c";
    }
    if (v == zendnn_aBC48b32c) {
        return "aBC48b32c";
    }
    if (v == zendnn_ABcd40a16b) {
        return "ABcd40a16b";
    }
    if (v == zendnn_ABcd40a32b) {
        return "ABcd40a32b";
    }
    if (v == zendnn_abCd32c) {
        return "abCd32c";
    }
    if (v == zendnn_abdCe32c) {
        return "abdCe32c";
    }
    if (v == zendnn_abdCE32c2e) {
        return "abdCE32c2e";
    }
    if (v == zendnn_BA16a16b2a) {
        return "BA16a16b2a";
    }
    if (v == zendnn_BA16a32b2a) {
        return "BA16a32b2a";
    }
    if (v == zendnn_BA16a48b2a) {
        return "BA16a48b2a";
    }
    if (v == zendnn_BA16a64b2a) {
        return "BA16a64b2a";
    }
    if (v == zendnn_BA16a16b4a) {
        return "BA16a16b4a";
    }
    if (v == zendnn_BA16a32b4a) {
        return "BA16a32b4a";
    }
    if (v == zendnn_BA16a48b4a) {
        return "BA16a48b4a";
    }
    if (v == zendnn_BA16a64b4a) {
        return "BA16a64b4a";
    }
    if (v == zendnn_ABcd8a2b) {
        return "ABcd8a2b";
    }
    if (v == zendnn_aBdeC16c16b2c) {
        return "aBdeC16c16b2c";
    }
    if (v == zendnn_aBdeC16c16b4c) {
        return "aBdeC16c16b4c";
    }
    if (v == zendnn_aBdefC16c16b2c) {
        return "aBdefC16c16b2c";
    }
    if (v == zendnn_AcB16b16a2b) {
        return "AcB16b16a2b";
    }
    if (v == zendnn_AcB16b16a4b) {
        return "AcB16b16a4b";
    }
    if (v == zendnn_AcdB16b16a2b) {
        return "AcdB16b16a2b";
    }
    if (v == zendnn_AcdB16b16a4b) {
        return "AcdB16b16a4b";
    }
    if (v == zendnn_AcdeB16b16a2b) {
        return "AcdeB16b16a2b";
    }
    if (v == zendnn_aBdefC16c16b4c) {
        return "aBdefC16c16b4c";
    }
    if (v == zendnn_AcdeB16b16a4b) {
        return "AcdeB16b16a4b";
    }
    if (v == zendnn_AcB16b32a2b) {
        return "AcB16b32a2b";
    }
    if (v == zendnn_AcB16b32a4b) {
        return "AcB16b32a4b";
    }
    if (v == zendnn_AcB16b48a2b) {
        return "AcB16b48a2b";
    }
    if (v == zendnn_AcB16b48a4b) {
        return "AcB16b48a4b";
    }
    if (v == zendnn_AcB16b64a2b) {
        return "AcB16b64a2b";
    }
    if (v == zendnn_AcB16b64a4b) {
        return "AcB16b64a4b";
    }
    if (v == zendnn_aBdC16c16b2c) {
        return "aBdC16c16b2c";
    }
    if (v == zendnn_aBdC16c16b4c) {
        return "aBdC16c16b4c";
    }
    if (v == zendnn_aBdC16c32b2c) {
        return "aBdC16c32b2c";
    }
    if (v == zendnn_aBdC16c32b4c) {
        return "aBdC16c32b4c";
    }
    if (v == zendnn_aBdC16c48b2c) {
        return "aBdC16c48b2c";
    }
    if (v == zendnn_aBdC16c48b4c) {
        return "aBdC16c48b4c";
    }
    if (v == zendnn_aBdC16c64b2c) {
        return "aBdC16c64b2c";
    }
    if (v == zendnn_aBdC16c64b4c) {
        return "aBdC16c64b4c";
    }
    if (v == zendnn_AcdB16b32a2b) {
        return "AcdB16b32a2b";
    }
    if (v == zendnn_AcdB16b32a4b) {
        return "AcdB16b32a4b";
    }
    if (v == zendnn_AcdB16b48a2b) {
        return "AcdB16b48a2b";
    }
    if (v == zendnn_AcdB16b48a4b) {
        return "AcdB16b48a4b";
    }
    if (v == zendnn_AcdB16b64a2b) {
        return "AcdB16b64a2b";
    }
    if (v == zendnn_AcdB16b64a4b) {
        return "AcdB16b64a4b";
    }
    if (v == zendnn_aBdeC16c32b2c) {
        return "aBdeC16c32b2c";
    }
    if (v == zendnn_aBdeC16c32b4c) {
        return "aBdeC16c32b4c";
    }
    if (v == zendnn_aBdeC16c48b2c) {
        return "aBdeC16c48b2c";
    }
    if (v == zendnn_aBdeC16c48b4c) {
        return "aBdeC16c48b4c";
    }
    if (v == zendnn_aBdeC16c64b2c) {
        return "aBdeC16c64b2c";
    }
    if (v == zendnn_aBdeC16c64b4c) {
        return "aBdeC16c64b4c";
    }
    if (v == zendnn_AcdeB16b32a2b) {
        return "AcdeB16b32a2b";
    }
    if (v == zendnn_AcdeB16b32a4b) {
        return "AcdeB16b32a4b";
    }
    if (v == zendnn_AcdeB16b48a2b) {
        return "AcdeB16b48a2b";
    }
    if (v == zendnn_AcdeB16b48a4b) {
        return "AcdeB16b48a4b";
    }
    if (v == zendnn_AcdeB16b64a2b) {
        return "AcdeB16b64a2b";
    }
    if (v == zendnn_AcdeB16b64a4b) {
        return "AcdeB16b64a4b";
    }
    if (v == zendnn_aBdefC16c32b2c) {
        return "aBdefC16c32b2c";
    }
    if (v == zendnn_aBdefC16c32b4c) {
        return "aBdefC16c32b4c";
    }
    if (v == zendnn_aBdefC16c48b2c) {
        return "aBdefC16c48b2c";
    }
    if (v == zendnn_aBdefC16c48b4c) {
        return "aBdefC16c48b4c";
    }
    if (v == zendnn_aBdefC16c64b2c) {
        return "aBdefC16c64b2c";
    }
    if (v == zendnn_aBdefC16c64b4c) {
        return "aBdefC16c64b4c";
    }
    if (v == zendnn_decbA16a) {
        return "decbA16a";
    }
    if (v == zendnn_ABc4a2b) {
        return "ABc4a2b";
    }
    if (v == zendnn_ABc8a2b) {
        return "ABc8a2b";
    }
    if (v == zendnn_aBCd8b2c) {
        return "aBCd8b2c";
    }
    if (v == zendnn_ABcde4a2b) {
        return "ABcde4a2b";
    }
    if (v == zendnn_ABcde8a2b) {
        return "ABcde8a2b";
    }
    if (v == zendnn_ABcde40a16b) {
        return "ABcde40a16b";
    }
    if (v == zendnn_ABcde40a32b) {
        return "ABcde40a32b";
    }
    if (v == zendnn_aBCde8b2c) {
        return "aBCde8b2c";
    }
    if (v == zendnn_ABcde4a8b8a2b) {
        return "ABcde4a8b8a2b";
    }
    if (v == zendnn_ABcd4a8b8a2b) {
        return "ABcd4a8b8a2b";
    }
    if (v == zendnn_ABc4a8b8a2b) {
        return "ABc4a8b8a2b";
    }
    if (v == zendnn_aBCdef4b8c8b2c) {
        return "aBCdef4b8c8b2c";
    }
    if (v == zendnn_aBCde4b8c8b2c) {
        return "aBCde4b8c8b2c";
    }
    if (v == zendnn_aBCd4b8c8b2c) {
        return "aBCd4b8c8b2c";
    }
    if (v == zendnn_BAcde4b8a8b2a) {
        return "BAcde4b8a8b2a";
    }
    if (v == zendnn_BAcd4b8a8b2a) {
        return "BAcd4b8a8b2a";
    }
    if (v == zendnn_BAc4b8a8b2a) {
        return "BAc4b8a8b2a";
    }
    if (v == zendnn_aCBdef4c8b8c2b) {
        return "aCBdef4c8b8c2b";
    }
    if (v == zendnn_aCBde4c8b8c2b) {
        return "aCBde4c8b8c2b";
    }
    if (v == zendnn_aCBd4c8b8c2b) {
        return "aCBd4c8b8c2b";
    }
    if (v == zendnn_aBCdef8b2c) {
        return "aBCdef8b2c";
    }
    if (v == zendnn_AB32a16b) {
        return "AB32a16b";
    }
    if (v == zendnn_AB32a32b) {
        return "AB32a32b";
    }
    if (v == zendnn_BA4b8a8b2a) {
        return "BA4b8a8b2a";
    }
    if (v == zendnn_BA4b8a8b4a) {
        return "BA4b8a8b4a";
    }
    if (v == zendnn_aBC32b16c) {
        return "aBC32b16c";
    }
    if (v == zendnn_aBC32b32c) {
        return "aBC32b32c";
    }
    if (v == zendnn_aCB4c8b8c2b) {
        return "aCB4c8b8c2b";
    }
    if (v == zendnn_aCB4c8b8c4b) {
        return "aCB4c8b8c4b";
    }
    if (v == zendnn_ABcd4a2b) {
        return "ABcd4a2b";
    }
    if (v == zendnn_ABc2b8a16b4a) {
        return "ABc2b8a16b4a";
    }
    if (v == zendnn_ABcd2b8a16b4a) {
        return "ABcd2b8a16b4a";
    }
    if (v == zendnn_ABcde2b8a16b4a) {
        return "ABcde2b8a16b4a";
    }
    if (v == zendnn_ABc2a8b16a4b) {
        return "ABc2a8b16a4b";
    }
    if (v == zendnn_ABc2a8b16a2b) {
        return "ABc2a8b16a2b";
    }
    if (v == zendnn_ABc2b32a8b) {
        return "ABc2b32a8b";
    }
    if (v == zendnn_ABcd2a8b16a4b) {
        return "ABcd2a8b16a4b";
    }
    if (v == zendnn_ABcd2a8b16a2b) {
        return "ABcd2a8b16a2b";
    }
    if (v == zendnn_aCBd2c8b16c2b) {
        return "aCBd2c8b16c2b";
    }
    if (v == zendnn_ABcd2b32a8b) {
        return "ABcd2b32a8b";
    }
    if (v == zendnn_aBCd2c8b16c2b) {
        return "aBCd2c8b16c2b";
    }
    if (v == zendnn_ABcde2a8b16a4b) {
        return "ABcde2a8b16a4b";
    }
    if (v == zendnn_ABcde2a8b16a2b) {
        return "ABcde2a8b16a2b";
    }
    if (v == zendnn_aCBde2c8b16c2b) {
        return "aCBde2c8b16c2b";
    }
    if (v == zendnn_ABcde2b32a8b) {
        return "ABcde2b32a8b";
    }
    if (v == zendnn_aBC2b8c16b2c) {
        return "aBC2b8c16b2c";
    }
    if (v == zendnn_aBCd2b8c16b2c) {
        return "aBCd2b8c16b2c";
    }
    if (v == zendnn_aBCde2b8c16b2c) {
        return "aBCde2b8c16b2c";
    }
    if (v == zendnn_aBCdef2b8c16b2c) {
        return "aBCdef2b8c16b2c";
    }
    if (v == zendnn_BAcde2b8a16b4a) {
        return "BAcde2b8a16b4a";
    }
    if (v == zendnn_BAcd2b8a16b4a) {
        return "BAcd2b8a16b4a";
    }
    if (v == zendnn_BAc2b8a16b4a) {
        return "BAc2b8a16b4a";
    }
    if (v == zendnn_BAcde2b8a16b2a) {
        return "BAcde2b8a16b2a";
    }
    if (v == zendnn_BAcd2b8a16b2a) {
        return "BAcd2b8a16b2a";
    }
    if (v == zendnn_BAc2b8a16b2a) {
        return "BAc2b8a16b2a";
    }
    if (v == zendnn_aBCde2c8b16c2b) {
        return "aBCde2c8b16c2b";
    }
    if (v == zendnn_aBCdef2c8b16c2b) {
        return "aBCdef2c8b16c2b";
    }
    if (v == zendnn_aCBdef2c8b16c2b) {
        return "aCBdef2c8b16c2b";
    }
    if (v == zendnn_aBCd2b8c16b4c) {
        return "aBCd2b8c16b4c";
    }
    if (v == zendnn_aBCde2b8c16b4c) {
        return "aBCde2b8c16b4c";
    }
    if (v == zendnn_BA4b8a16b2a) {
        return "BA4b8a16b2a";
    }
    if (v == zendnn_BA4b8a16b4a) {
        return "BA4b8a16b4a";
    }
    if (v == zendnn_aCB4c8b16c2b) {
        return "aCB4c8b16c2b";
    }
    if (v == zendnn_aCB4c8b16c4b) {
        return "aCB4c8b16c4b";
    }
    if (v == zendnn_BA16a16b) {
        return "BA16a16b";
    }
    if (v == zendnn_BA16a32b) {
        return "BA16a32b";
    }
    if (v == zendnn_BA16a48b) {
        return "BA16a48b";
    }
    if (v == zendnn_BA16a64b) {
        return "BA16a64b";
    }
    if (v == zendnn_aCB16c2b) {
        return "aCB16c2b";
    }
    if (v == zendnn_aCB16c4b) {
        return "aCB16c4b";
    }
    if (v == zendnn_BA16b2a) {
        return "BA16b2a";
    }
    if (v == zendnn_BA16b4a) {
        return "BA16b4a";
    }
    if (v == zendnn_aBC16b16c) {
        return "aBC16b16c";
    }
    if (v == zendnn_aBC16b32c) {
        return "aBC16b32c";
    }
    if (v == zendnn_AB16a16b) {
        return "AB16a16b";
    }
    if (v == zendnn_AB16a32b) {
        return "AB16a32b";
    }
    if (v == zendnn_adbc) {
        return "adbc";
    }
    if (v == zendnn_ABcde16a16b2a) {
        return "ABcde16a16b2a";
    }
    if (v == zendnn_aBCdef16b16c2b) {
        return "aBCdef16b16c2b";
    }
    if (v == zendnn_format_tag_last) {
        return "format_tag_last";
    }
    if (v == zendnn_x) {
        return "x";
    }
    if (v == zendnn_nc) {
        return "nc";
    }
    if (v == zendnn_cn) {
        return "cn";
    }
    if (v == zendnn_tn) {
        return "tn";
    }
    if (v == zendnn_nt) {
        return "nt";
    }
    if (v == zendnn_ncw) {
        return "ncw";
    }
    if (v == zendnn_nwc) {
        return "nwc";
    }
    if (v == zendnn_nchw) {
        return "nchw";
    }
    if (v == zendnn_nhwc) {
        return "nhwc";
    }
    if (v == zendnn_chwn) {
        return "chwn";
    }
    if (v == zendnn_hwcn) {
        return "hwcn";
    }
    if (v == zendnn_ncdhw) {
        return "ncdhw";
    }
    if (v == zendnn_ndhwc) {
        return "ndhwc";
    }
    if (v == zendnn_oi) {
        return "oi";
    }
    if (v == zendnn_io) {
        return "io";
    }
    if (v == zendnn_oiw) {
        return "oiw";
    }
    if (v == zendnn_owi) {
        return "owi";
    }
    if (v == zendnn_wio) {
        return "wio";
    }
    if (v == zendnn_iwo) {
        return "iwo";
    }
    if (v == zendnn_oihw) {
        return "oihw";
    }
    if (v == zendnn_hwio) {
        return "hwio";
    }
    if (v == zendnn_ohwi) {
        return "ohwi";
    }
    if (v == zendnn_ihwo) {
        return "ihwo";
    }
    if (v == zendnn_iohw) {
        return "iohw";
    }
    if (v == zendnn_oidhw) {
        return "oidhw";
    }
    if (v == zendnn_iodhw) {
        return "iodhw";
    }
    if (v == zendnn_dhwio) {
        return "dhwio";
    }
    if (v == zendnn_odhwi) {
        return "odhwi";
    }
    if (v == zendnn_idhwo) {
        return "idhwo";
    }
    if (v == zendnn_goiw) {
        return "goiw";
    }
    if (v == zendnn_gowi) {
        return "gowi";
    }
    if (v == zendnn_wigo) {
        return "wigo";
    }
    if (v == zendnn_goihw) {
        return "goihw";
    }
    if (v == zendnn_gohwi) {
        return "gohwi";
    }
    if (v == zendnn_hwigo) {
        return "hwigo";
    }
    if (v == zendnn_giohw) {
        return "giohw";
    }
    if (v == zendnn_goidhw) {
        return "goidhw";
    }
    if (v == zendnn_godhwi) {
        return "godhwi";
    }
    if (v == zendnn_giodhw) {
        return "giodhw";
    }
    if (v == zendnn_dhwigo) {
        return "dhwigo";
    }
    if (v == zendnn_tnc) {
        return "tnc";
    }
    if (v == zendnn_ntc) {
        return "ntc";
    }
    if (v == zendnn_ldnc) {
        return "ldnc";
    }
    if (v == zendnn_ldigo) {
        return "ldigo";
    }
    if (v == zendnn_ldgoi) {
        return "ldgoi";
    }
    if (v == zendnn_ldio) {
        return "ldio";
    }
    if (v == zendnn_ldoi) {
        return "ldoi";
    }
    if (v == zendnn_ldgo) {
        return "ldgo";
    }
    if (v == zendnn_ldOi32o) {
        return "ldOi32o";
    }
    if (v == zendnn_ldOI32o4i) {
        return "ldOI32o4i";
    }
    if (v == zendnn_ldIo32i) {
        return "ldIo32i";
    }
    if (v == zendnn_ldgOi32o) {
        return "ldgOi32o";
    }
    if (v == zendnn_ldgOI32o2i) {
        return "ldgOI32o2i";
    }
    if (v == zendnn_ldgOI32o4i) {
        return "ldgOI32o4i";
    }
    if (v == zendnn_ldgIo32i) {
        return "ldgIo32i";
    }
    if (v == zendnn_ldgIO32i2o) {
        return "ldgIO32i2o";
    }
    if (v == zendnn_nCdhw32c) {
        return "nCdhw32c";
    }
    if (v == zendnn_nCdhw16c) {
        return "nCdhw16c";
    }
    if (v == zendnn_nCdhw4c) {
        return "nCdhw4c";
    }
    if (v == zendnn_nCdhw8c) {
        return "nCdhw8c";
    }
    if (v == zendnn_nChw32c) {
        return "nChw32c";
    }
    if (v == zendnn_nChw16c) {
        return "nChw16c";
    }
    if (v == zendnn_nChw4c) {
        return "nChw4c";
    }
    if (v == zendnn_nChw8c) {
        return "nChw8c";
    }
    if (v == zendnn_nCw32c) {
        return "nCw32c";
    }
    if (v == zendnn_nCw16c) {
        return "nCw16c";
    }
    if (v == zendnn_nCw4c) {
        return "nCw4c";
    }
    if (v == zendnn_nCw8c) {
        return "nCw8c";
    }
    if (v == zendnn_NCw16n16c) {
        return "NCw16n16c";
    }
    if (v == zendnn_NCdhw16n16c) {
        return "NCdhw16n16c";
    }
    if (v == zendnn_NChw16n16c) {
        return "NChw16n16c";
    }
    if (v == zendnn_NCw32n16c) {
        return "NCw32n16c";
    }
    if (v == zendnn_NChw32n16c) {
        return "NChw32n16c";
    }
    if (v == zendnn_NCdhw32n16c) {
        return "NCdhw32n16c";
    }
    if (v == zendnn_NCw32n32c) {
        return "NCw32n32c";
    }
    if (v == zendnn_NChw32n32c) {
        return "NChw32n32c";
    }
    if (v == zendnn_NCdhw32n32c) {
        return "NCdhw32n32c";
    }
    if (v == zendnn_OI16i16o) {
        return "OI16i16o";
    }
    if (v == zendnn_OI16i32o) {
        return "OI16i32o";
    }
    if (v == zendnn_OI16i64o) {
        return "OI16i64o";
    }
    if (v == zendnn_OI8i16o2i) {
        return "OI8i16o2i";
    }
    if (v == zendnn_OI8i32o2i) {
        return "OI8i32o2i";
    }
    if (v == zendnn_OI8i64o2i) {
        return "OI8i64o2i";
    }
    if (v == zendnn_OI4i16o4i) {
        return "OI4i16o4i";
    }
    if (v == zendnn_OI4i32o4i) {
        return "OI4i32o4i";
    }
    if (v == zendnn_OI4i64o4i) {
        return "OI4i64o4i";
    }
    if (v == zendnn_OI16i16o4i) {
        return "OI16i16o4i";
    }
    if (v == zendnn_IOw16o16i) {
        return "IOw16o16i";
    }
    if (v == zendnn_IOw16i16o) {
        return "IOw16i16o";
    }
    if (v == zendnn_OIw16i16o) {
        return "OIw16i16o";
    }
    if (v == zendnn_OIw16i32o) {
        return "OIw16i32o";
    }
    if (v == zendnn_OIw16i64o) {
        return "OIw16i64o";
    }
    if (v == zendnn_OIw16o16i) {
        return "OIw16o16i";
    }
    if (v == zendnn_Oiw16o) {
        return "Oiw16o";
    }
    if (v == zendnn_OIw4i16o4i) {
        return "OIw4i16o4i";
    }
    if (v == zendnn_OIw4i32o4i) {
        return "OIw4i32o4i";
    }
    if (v == zendnn_OIw4i64o4i) {
        return "OIw4i64o4i";
    }
    if (v == zendnn_OIw2i8o4i) {
        return "OIw2i8o4i";
    }
    if (v == zendnn_OIw16i16o4i) {
        return "OIw16i16o4i";
    }
    if (v == zendnn_OIw16i16o2i) {
        return "OIw16i16o2i";
    }
    if (v == zendnn_OIw16o16i2o) {
        return "OIw16o16i2o";
    }
    if (v == zendnn_OIw4i4o) {
        return "OIw4i4o";
    }
    if (v == zendnn_OIw4o4i) {
        return "OIw4o4i";
    }
    if (v == zendnn_Oiw4o) {
        return "Oiw4o";
    }
    if (v == zendnn_OIw8i16o2i) {
        return "OIw8i16o2i";
    }
    if (v == zendnn_OIw8i32o2i) {
        return "OIw8i32o2i";
    }
    if (v == zendnn_OIw8i64o2i) {
        return "OIw8i64o2i";
    }
    if (v == zendnn_OIw8i8o) {
        return "OIw8i8o";
    }
    if (v == zendnn_OIw8o16i2o) {
        return "OIw8o16i2o";
    }
    if (v == zendnn_IOw8o16i2o) {
        return "IOw8o16i2o";
    }
    if (v == zendnn_OIw8o8i) {
        return "OIw8o8i";
    }
    if (v == zendnn_OIw8o4i) {
        return "OIw8o4i";
    }
    if (v == zendnn_Owi16o) {
        return "Owi16o";
    }
    if (v == zendnn_OwI16o2i) {
        return "OwI16o2i";
    }
    if (v == zendnn_OwI16o4i) {
        return "OwI16o4i";
    }
    if (v == zendnn_Owi4o) {
        return "Owi4o";
    }
    if (v == zendnn_Owi8o) {
        return "Owi8o";
    }
    if (v == zendnn_IOhw16i16o) {
        return "IOhw16i16o";
    }
    if (v == zendnn_IOhw16o16i) {
        return "IOhw16o16i";
    }
    if (v == zendnn_Ohwi16o) {
        return "Ohwi16o";
    }
    if (v == zendnn_OhwI16o2i) {
        return "OhwI16o2i";
    }
    if (v == zendnn_OhwI16o4i) {
        return "OhwI16o4i";
    }
    if (v == zendnn_Ohwi32o) {
        return "Ohwi32o";
    }
    if (v == zendnn_Ohwi4o) {
        return "Ohwi4o";
    }
    if (v == zendnn_Ohwi8o) {
        return "Ohwi8o";
    }
    if (v == zendnn_OIhw16i16o) {
        return "OIhw16i16o";
    }
    if (v == zendnn_OIhw16i32o) {
        return "OIhw16i32o";
    }
    if (v == zendnn_OIhw16i64o) {
        return "OIhw16i64o";
    }
    if (v == zendnn_OIhw16o16i) {
        return "OIhw16o16i";
    }
    if (v == zendnn_Oihw16o) {
        return "Oihw16o";
    }
    if (v == zendnn_OIhw4i16o4i) {
        return "OIhw4i16o4i";
    }
    if (v == zendnn_OIhw4i32o4i) {
        return "OIhw4i32o4i";
    }
    if (v == zendnn_OIhw4i64o4i) {
        return "OIhw4i64o4i";
    }
    if (v == zendnn_OIhw16i16o4i) {
        return "OIhw16i16o4i";
    }
    if (v == zendnn_OIhw16i16o2i) {
        return "OIhw16i16o2i";
    }
    if (v == zendnn_OIhw16o16i2o) {
        return "OIhw16o16i2o";
    }
    if (v == zendnn_OIhw4i4o) {
        return "OIhw4i4o";
    }
    if (v == zendnn_OIhw4o4i) {
        return "OIhw4o4i";
    }
    if (v == zendnn_Oihw4o) {
        return "Oihw4o";
    }
    if (v == zendnn_OIhw8i16o2i) {
        return "OIhw8i16o2i";
    }
    if (v == zendnn_OIhw8i32o2i) {
        return "OIhw8i32o2i";
    }
    if (v == zendnn_OIhw8i64o2i) {
        return "OIhw8i64o2i";
    }
    if (v == zendnn_OIhw8i8o) {
        return "OIhw8i8o";
    }
    if (v == zendnn_OIhw8o16i2o) {
        return "OIhw8o16i2o";
    }
    if (v == zendnn_OIhw2i8o4i) {
        return "OIhw2i8o4i";
    }
    if (v == zendnn_IOhw8o16i2o) {
        return "IOhw8o16i2o";
    }
    if (v == zendnn_OIhw8o8i) {
        return "OIhw8o8i";
    }
    if (v == zendnn_OIhw8o4i) {
        return "OIhw8o4i";
    }
    if (v == zendnn_Owhi16o) {
        return "Owhi16o";
    }
    if (v == zendnn_Odhwi16o) {
        return "Odhwi16o";
    }
    if (v == zendnn_OdhwI16o2i) {
        return "OdhwI16o2i";
    }
    if (v == zendnn_OdhwI16o4i) {
        return "OdhwI16o4i";
    }
    if (v == zendnn_Odhwi4o) {
        return "Odhwi4o";
    }
    if (v == zendnn_Odhwi8o) {
        return "Odhwi8o";
    }
    if (v == zendnn_OIdhw16i16o) {
        return "OIdhw16i16o";
    }
    if (v == zendnn_OIdhw16i32o) {
        return "OIdhw16i32o";
    }
    if (v == zendnn_OIdhw16i64o) {
        return "OIdhw16i64o";
    }
    if (v == zendnn_OIdhw16o16i) {
        return "OIdhw16o16i";
    }
    if (v == zendnn_Oidhw16o) {
        return "Oidhw16o";
    }
    if (v == zendnn_OIdhw4i4o) {
        return "OIdhw4i4o";
    }
    if (v == zendnn_OIdhw4o4i) {
        return "OIdhw4o4i";
    }
    if (v == zendnn_Oidhw4o) {
        return "Oidhw4o";
    }
    if (v == zendnn_OIdhw8i16o2i) {
        return "OIdhw8i16o2i";
    }
    if (v == zendnn_OIdhw8i32o2i) {
        return "OIdhw8i32o2i";
    }
    if (v == zendnn_OIdhw8i64o2i) {
        return "OIdhw8i64o2i";
    }
    if (v == zendnn_OIdhw8i8o) {
        return "OIdhw8i8o";
    }
    if (v == zendnn_OIdhw8o16i2o) {
        return "OIdhw8o16i2o";
    }
    if (v == zendnn_IOdhw8o16i2o) {
        return "IOdhw8o16i2o";
    }
    if (v == zendnn_OIdhw4i16o4i) {
        return "OIdhw4i16o4i";
    }
    if (v == zendnn_OIdhw4i32o4i) {
        return "OIdhw4i32o4i";
    }
    if (v == zendnn_OIdhw4i64o4i) {
        return "OIdhw4i64o4i";
    }
    if (v == zendnn_OIdhw16i16o4i) {
        return "OIdhw16i16o4i";
    }
    if (v == zendnn_OIdhw16i16o2i) {
        return "OIdhw16i16o2i";
    }
    if (v == zendnn_OIdhw2i8o4i) {
        return "OIdhw2i8o4i";
    }
    if (v == zendnn_OIdhw8o8i) {
        return "OIdhw8o8i";
    }
    if (v == zendnn_OIdhw8o4i) {
        return "OIdhw8o4i";
    }
    if (v == zendnn_IOdhw16i16o) {
        return "IOdhw16i16o";
    }
    if (v == zendnn_OIdhw4o8i8o4i) {
        return "OIdhw4o8i8o4i";
    }
    if (v == zendnn_IOdhw16o16i) {
        return "IOdhw16o16i";
    }
    if (v == zendnn_OIdhw16o16i2o) {
        return "OIdhw16o16i2o";
    }
    if (v == zendnn_Goiw16g) {
        return "Goiw16g";
    }
    if (v == zendnn_Goiw8g) {
        return "Goiw8g";
    }
    if (v == zendnn_Goiw4g) {
        return "Goiw4g";
    }
    if (v == zendnn_gIOw16o16i) {
        return "gIOw16o16i";
    }
    if (v == zendnn_gIOw16i16o) {
        return "gIOw16i16o";
    }
    if (v == zendnn_gOIw16i16o) {
        return "gOIw16i16o";
    }
    if (v == zendnn_gOIw16o16i) {
        return "gOIw16o16i";
    }
    if (v == zendnn_gOiw16o) {
        return "gOiw16o";
    }
    if (v == zendnn_gOIw4i16o4i) {
        return "gOIw4i16o4i";
    }
    if (v == zendnn_gOIw2i8o4i) {
        return "gOIw2i8o4i";
    }
    if (v == zendnn_gOIw16i16o4i) {
        return "gOIw16i16o4i";
    }
    if (v == zendnn_gOIw16i16o2i) {
        return "gOIw16i16o2i";
    }
    if (v == zendnn_gOIw16o16i2o) {
        return "gOIw16o16i2o";
    }
    if (v == zendnn_gOIw4i4o) {
        return "gOIw4i4o";
    }
    if (v == zendnn_gOIw4o4i) {
        return "gOIw4o4i";
    }
    if (v == zendnn_gOiw4o) {
        return "gOiw4o";
    }
    if (v == zendnn_gOIw8i16o2i) {
        return "gOIw8i16o2i";
    }
    if (v == zendnn_gOIw8i8o) {
        return "gOIw8i8o";
    }
    if (v == zendnn_gOIw8o16i2o) {
        return "gOIw8o16i2o";
    }
    if (v == zendnn_gIOw8o16i2o) {
        return "gIOw8o16i2o";
    }
    if (v == zendnn_gOIw8o8i) {
        return "gOIw8o8i";
    }
    if (v == zendnn_gOIw8o4i) {
        return "gOIw8o4i";
    }
    if (v == zendnn_gOwi16o) {
        return "gOwi16o";
    }
    if (v == zendnn_gOwI16o2i) {
        return "gOwI16o2i";
    }
    if (v == zendnn_gOwI16o4i) {
        return "gOwI16o4i";
    }
    if (v == zendnn_gOwi4o) {
        return "gOwi4o";
    }
    if (v == zendnn_gOwi8o) {
        return "gOwi8o";
    }
    if (v == zendnn_Goiw32g) {
        return "Goiw32g";
    }
    if (v == zendnn_gOIw2i4o2i) {
        return "gOIw2i4o2i";
    }
    if (v == zendnn_gOIw2o4i2o) {
        return "gOIw2o4i2o";
    }
    if (v == zendnn_gOIw4i8o2i) {
        return "gOIw4i8o2i";
    }
    if (v == zendnn_gOIw4o8i2o) {
        return "gOIw4o8i2o";
    }
    if (v == zendnn_gIOhw16i16o) {
        return "gIOhw16i16o";
    }
    if (v == zendnn_gIOhw16o16i) {
        return "gIOhw16o16i";
    }
    if (v == zendnn_gOhwi16o) {
        return "gOhwi16o";
    }
    if (v == zendnn_gOhwI16o2i) {
        return "gOhwI16o2i";
    }
    if (v == zendnn_gOhwI16o4i) {
        return "gOhwI16o4i";
    }
    if (v == zendnn_gOhwi32o) {
        return "gOhwi32o";
    }
    if (v == zendnn_gOhwi4o) {
        return "gOhwi4o";
    }
    if (v == zendnn_gOhwi8o) {
        return "gOhwi8o";
    }
    if (v == zendnn_Goihw16g) {
        return "Goihw16g";
    }
    if (v == zendnn_gOIhw16i16o) {
        return "gOIhw16i16o";
    }
    if (v == zendnn_gOIhw16o16i) {
        return "gOIhw16o16i";
    }
    if (v == zendnn_gOihw16o) {
        return "gOihw16o";
    }
    if (v == zendnn_gOIhw2i8o4i) {
        return "gOIhw2i8o4i";
    }
    if (v == zendnn_gOIhw4i16o4i) {
        return "gOIhw4i16o4i";
    }
    if (v == zendnn_gOIhw16i16o4i) {
        return "gOIhw16i16o4i";
    }
    if (v == zendnn_gOIhw16i16o2i) {
        return "gOIhw16i16o2i";
    }
    if (v == zendnn_gOIhw16o16i2o) {
        return "gOIhw16o16i2o";
    }
    if (v == zendnn_gOIhw4i4o) {
        return "gOIhw4i4o";
    }
    if (v == zendnn_gOIhw4o4i) {
        return "gOIhw4o4i";
    }
    if (v == zendnn_gOihw4o) {
        return "gOihw4o";
    }
    if (v == zendnn_Goihw8g) {
        return "Goihw8g";
    }
    if (v == zendnn_Goihw4g) {
        return "Goihw4g";
    }
    if (v == zendnn_gOIhw8i16o2i) {
        return "gOIhw8i16o2i";
    }
    if (v == zendnn_gOIhw8i8o) {
        return "gOIhw8i8o";
    }
    if (v == zendnn_gOIhw8o16i2o) {
        return "gOIhw8o16i2o";
    }
    if (v == zendnn_gIOhw8o16i2o) {
        return "gIOhw8o16i2o";
    }
    if (v == zendnn_gOIhw8o8i) {
        return "gOIhw8o8i";
    }
    if (v == zendnn_gOIhw8o4i) {
        return "gOIhw8o4i";
    }
    if (v == zendnn_Goihw32g) {
        return "Goihw32g";
    }
    if (v == zendnn_gOwhi16o) {
        return "gOwhi16o";
    }
    if (v == zendnn_OIw4o8i8o4i) {
        return "OIw4o8i8o4i";
    }
    if (v == zendnn_OIhw4o8i8o4i) {
        return "OIhw4o8i8o4i";
    }
    if (v == zendnn_IOw4i8o8i4o) {
        return "IOw4i8o8i4o";
    }
    if (v == zendnn_IOhw4i8o8i4o) {
        return "IOhw4i8o8i4o";
    }
    if (v == zendnn_IOdhw4i8o8i4o) {
        return "IOdhw4i8o8i4o";
    }
    if (v == zendnn_OIhw2o8i8o2i) {
        return "OIhw2o8i8o2i";
    }
    if (v == zendnn_gOIw4o8i8o4i) {
        return "gOIw4o8i8o4i";
    }
    if (v == zendnn_gOIhw4o8i8o4i) {
        return "gOIhw4o8i8o4i";
    }
    if (v == zendnn_gOIdhw4o8i8o4i) {
        return "gOIdhw4o8i8o4i";
    }
    if (v == zendnn_gIOw4i8o8i4o) {
        return "gIOw4i8o8i4o";
    }
    if (v == zendnn_gIOhw4i8o8i4o) {
        return "gIOhw4i8o8i4o";
    }
    if (v == zendnn_gIOdhw4i8o8i4o) {
        return "gIOdhw4i8o8i4o";
    }
    if (v == zendnn_gOIhw2o8i8o2i) {
        return "gOIhw2o8i8o2i";
    }
    if (v == zendnn_gOIhw2i4o2i) {
        return "gOIhw2i4o2i";
    }
    if (v == zendnn_gOIhw2o4i2o) {
        return "gOIhw2o4i2o";
    }
    if (v == zendnn_gOIhw4i8o2i) {
        return "gOIhw4i8o2i";
    }
    if (v == zendnn_gOIhw4o8i2o) {
        return "gOIhw4o8i2o";
    }
    if (v == zendnn_gIOdhw16i16o) {
        return "gIOdhw16i16o";
    }
    if (v == zendnn_gIOdhw16o16i) {
        return "gIOdhw16o16i";
    }
    if (v == zendnn_gOdhwi16o) {
        return "gOdhwi16o";
    }
    if (v == zendnn_gOdhwI16o2i) {
        return "gOdhwI16o2i";
    }
    if (v == zendnn_gOdhwI16o4i) {
        return "gOdhwI16o4i";
    }
    if (v == zendnn_gOdhwi4o) {
        return "gOdhwi4o";
    }
    if (v == zendnn_gOdhwi8o) {
        return "gOdhwi8o";
    }
    if (v == zendnn_gOIdhw16i16o) {
        return "gOIdhw16i16o";
    }
    if (v == zendnn_gOIdhw4i16o4i) {
        return "gOIdhw4i16o4i";
    }
    if (v == zendnn_gOIdhw16i16o4i) {
        return "gOIdhw16i16o4i";
    }
    if (v == zendnn_gOIdhw2i8o4i) {
        return "gOIdhw2i8o4i";
    }
    if (v == zendnn_gOIdhw16i16o2i) {
        return "gOIdhw16i16o2i";
    }
    if (v == zendnn_gOIdhw16o16i) {
        return "gOIdhw16o16i";
    }
    if (v == zendnn_gOIdhw16o16i2o) {
        return "gOIdhw16o16i2o";
    }
    if (v == zendnn_gOidhw16o) {
        return "gOidhw16o";
    }
    if (v == zendnn_gOIdhw4i4o) {
        return "gOIdhw4i4o";
    }
    if (v == zendnn_gOIdhw4o4i) {
        return "gOIdhw4o4i";
    }
    if (v == zendnn_gOidhw4o) {
        return "gOidhw4o";
    }
    if (v == zendnn_gOIdhw8i16o2i) {
        return "gOIdhw8i16o2i";
    }
    if (v == zendnn_gOIdhw8i8o) {
        return "gOIdhw8i8o";
    }
    if (v == zendnn_gOIdhw8o16i2o) {
        return "gOIdhw8o16i2o";
    }
    if (v == zendnn_gIOdhw8o16i2o) {
        return "gIOdhw8o16i2o";
    }
    if (v == zendnn_gOIdhw8o8i) {
        return "gOIdhw8o8i";
    }
    if (v == zendnn_gOIdhw8o4i) {
        return "gOIdhw8o4i";
    }
    if (v == zendnn_Goidhw16g) {
        return "Goidhw16g";
    }
    if (v == zendnn_Goidhw32g) {
        return "Goidhw32g";
    }
    if (v == zendnn_gOIdhw2i4o2i) {
        return "gOIdhw2i4o2i";
    }
    if (v == zendnn_gOIdhw4i8o2i) {
        return "gOIdhw4i8o2i";
    }
    if (v == zendnn_gOIdhw2o4i2o) {
        return "gOIdhw2o4i2o";
    }
    if (v == zendnn_gOIdhw4o8i2o) {
        return "gOIdhw4o8i2o";
    }
    if (v == zendnn_Owi32o) {
        return "Owi32o";
    }
    if (v == zendnn_OwI32o2i) {
        return "OwI32o2i";
    }
    if (v == zendnn_OwI32o4i) {
        return "OwI32o4i";
    }
    if (v == zendnn_Owi48o) {
        return "Owi48o";
    }
    if (v == zendnn_OwI48o2i) {
        return "OwI48o2i";
    }
    if (v == zendnn_OwI48o4i) {
        return "OwI48o4i";
    }
    if (v == zendnn_Owi64o) {
        return "Owi64o";
    }
    if (v == zendnn_OwI64o2i) {
        return "OwI64o2i";
    }
    if (v == zendnn_OwI64o4i) {
        return "OwI64o4i";
    }
    if (v == zendnn_wIo2i) {
        return "wIo2i";
    }
    if (v == zendnn_wIo4i) {
        return "wIo4i";
    }
    if (v == zendnn_gOwi32o) {
        return "gOwi32o";
    }
    if (v == zendnn_gOwI32o2i) {
        return "gOwI32o2i";
    }
    if (v == zendnn_gOwI32o4i) {
        return "gOwI32o4i";
    }
    if (v == zendnn_gOwi48o) {
        return "gOwi48o";
    }
    if (v == zendnn_gOwI48o2i) {
        return "gOwI48o2i";
    }
    if (v == zendnn_gOwI48o4i) {
        return "gOwI48o4i";
    }
    if (v == zendnn_gOwi64o) {
        return "gOwi64o";
    }
    if (v == zendnn_gOwI64o2i) {
        return "gOwI64o2i";
    }
    if (v == zendnn_gOwI64o4i) {
        return "gOwI64o4i";
    }
    if (v == zendnn_gwio) {
        return "gwio";
    }
    if (v == zendnn_gwIo2i) {
        return "gwIo2i";
    }
    if (v == zendnn_gwIo4i) {
        return "gwIo4i";
    }
    if (v == zendnn_OhwI32o) {
        return "OhwI32o";
    }
    if (v == zendnn_OhwI32o2i) {
        return "OhwI32o2i";
    }
    if (v == zendnn_OhwI32o4i) {
        return "OhwI32o4i";
    }
    if (v == zendnn_Ohwi48o) {
        return "Ohwi48o";
    }
    if (v == zendnn_OhwI48o2i) {
        return "OhwI48o2i";
    }
    if (v == zendnn_OhwI48o4i) {
        return "OhwI48o4i";
    }
    if (v == zendnn_Ohwi64o) {
        return "Ohwi64o";
    }
    if (v == zendnn_OhwI64o2i) {
        return "OhwI64o2i";
    }
    if (v == zendnn_OhwI64o4i) {
        return "OhwI64o4i";
    }
    if (v == zendnn_hwIo2i) {
        return "hwIo2i";
    }
    if (v == zendnn_hwIo4i) {
        return "hwIo4i";
    }
    if (v == zendnn_gOhwI32o) {
        return "gOhwI32o";
    }
    if (v == zendnn_gOhwI32o2i) {
        return "gOhwI32o2i";
    }
    if (v == zendnn_gOhwI32o4i) {
        return "gOhwI32o4i";
    }
    if (v == zendnn_gOhwi48o) {
        return "gOhwi48o";
    }
    if (v == zendnn_gOhwI48o2i) {
        return "gOhwI48o2i";
    }
    if (v == zendnn_gOhwI48o4i) {
        return "gOhwI48o4i";
    }
    if (v == zendnn_gOhwi64o) {
        return "gOhwi64o";
    }
    if (v == zendnn_gOhwI64o2i) {
        return "gOhwI64o2i";
    }
    if (v == zendnn_gOhwI64o4i) {
        return "gOhwI64o4i";
    }
    if (v == zendnn_ghwio) {
        return "ghwio";
    }
    if (v == zendnn_ghwIo2i) {
        return "ghwIo2i";
    }
    if (v == zendnn_ghwIo4i) {
        return "ghwIo4i";
    }
    if (v == zendnn_Odhwi32o) {
        return "Odhwi32o";
    }
    if (v == zendnn_OdhwI32o2i) {
        return "OdhwI32o2i";
    }
    if (v == zendnn_OdhwI32o4i) {
        return "OdhwI32o4i";
    }
    if (v == zendnn_Odhwi48o) {
        return "Odhwi48o";
    }
    if (v == zendnn_OdhwI48o2i) {
        return "OdhwI48o2i";
    }
    if (v == zendnn_OdhwI48o4i) {
        return "OdhwI48o4i";
    }
    if (v == zendnn_Odhwi64o) {
        return "Odhwi64o";
    }
    if (v == zendnn_OdhwI64o2i) {
        return "OdhwI64o2i";
    }
    if (v == zendnn_OdhwI64o4i) {
        return "OdhwI64o4i";
    }
    if (v == zendnn_dhwIo2i) {
        return "dhwIo2i";
    }
    if (v == zendnn_dhwIo4i) {
        return "dhwIo4i";
    }
    if (v == zendnn_gOdhwi32o) {
        return "gOdhwi32o";
    }
    if (v == zendnn_gOdhwI32o2i) {
        return "gOdhwI32o2i";
    }
    if (v == zendnn_gOdhwI32o4i) {
        return "gOdhwI32o4i";
    }
    if (v == zendnn_gOdhwi48o) {
        return "gOdhwi48o";
    }
    if (v == zendnn_gOdhwI48o2i) {
        return "gOdhwI48o2i";
    }
    if (v == zendnn_gOdhwI48o4i) {
        return "gOdhwI48o4i";
    }
    if (v == zendnn_gOdhwi64o) {
        return "gOdhwi64o";
    }
    if (v == zendnn_gOdhwI64o2i) {
        return "gOdhwI64o2i";
    }
    if (v == zendnn_gOdhwI64o4i) {
        return "gOdhwI64o4i";
    }
    if (v == zendnn_gdhwio) {
        return "gdhwio";
    }
    if (v == zendnn_gdhwIo2i) {
        return "gdhwIo2i";
    }
    if (v == zendnn_gdhwIo4i) {
        return "gdhwIo4i";
    }
    if (v == zendnn_OI16i32o4i) {
        return "OI16i32o4i";
    }
    if (v == zendnn_OI16i48o4i) {
        return "OI16i48o4i";
    }
    if (v == zendnn_OI16i64o4i) {
        return "OI16i64o4i";
    }
    if (v == zendnn_OI16i16o2i) {
        return "OI16i16o2i";
    }
    if (v == zendnn_OI16i32o2i) {
        return "OI16i32o2i";
    }
    if (v == zendnn_OI16i48o2i) {
        return "OI16i48o2i";
    }
    if (v == zendnn_OI16i64o2i) {
        return "OI16i64o2i";
    }
    if (v == zendnn_OIw16i32o4i) {
        return "OIw16i32o4i";
    }
    if (v == zendnn_OIw16i48o4i) {
        return "OIw16i48o4i";
    }
    if (v == zendnn_OIw16i64o4i) {
        return "OIw16i64o4i";
    }
    if (v == zendnn_OIw16i32o2i) {
        return "OIw16i32o2i";
    }
    if (v == zendnn_OIw16i48o2i) {
        return "OIw16i48o2i";
    }
    if (v == zendnn_OIw16i64o2i) {
        return "OIw16i64o2i";
    }
    if (v == zendnn_OIhw16i32o4i) {
        return "OIhw16i32o4i";
    }
    if (v == zendnn_OIhw16i48o4i) {
        return "OIhw16i48o4i";
    }
    if (v == zendnn_OIhw16i64o4i) {
        return "OIhw16i64o4i";
    }
    if (v == zendnn_OIhw16i32o2i) {
        return "OIhw16i32o2i";
    }
    if (v == zendnn_OIhw16i48o2i) {
        return "OIhw16i48o2i";
    }
    if (v == zendnn_OIhw16i64o2i) {
        return "OIhw16i64o2i";
    }
    if (v == zendnn_OIdhw16i32o4i) {
        return "OIdhw16i32o4i";
    }
    if (v == zendnn_OIdhw16i48o4i) {
        return "OIdhw16i48o4i";
    }
    if (v == zendnn_OIdhw16i64o4i) {
        return "OIdhw16i64o4i";
    }
    if (v == zendnn_OIdhw16i32o2i) {
        return "OIdhw16i32o2i";
    }
    if (v == zendnn_OIdhw16i48o2i) {
        return "OIdhw16i48o2i";
    }
    if (v == zendnn_OIdhw16i64o2i) {
        return "OIdhw16i64o2i";
    }
    if (v == zendnn_OwI16i16o2i) {
        return "OwI16i16o2i";
    }
    if (v == zendnn_OwI16i16o4i) {
        return "OwI16i16o4i";
    }
    if (v == zendnn_OhwI16i16o2i) {
        return "OhwI16i16o2i";
    }
    if (v == zendnn_OhwI16i16o4i) {
        return "OhwI16i16o4i";
    }
    if (v == zendnn_OdhwI16i16o2i) {
        return "OdhwI16i16o2i";
    }
    if (v == zendnn_OdhwI16i16o4i) {
        return "OdhwI16i16o4i";
    }
    if (v == zendnn_gOwI16i16o2i) {
        return "gOwI16i16o2i";
    }
    if (v == zendnn_gOwI16i16o4i) {
        return "gOwI16i16o4i";
    }
    if (v == zendnn_gOhwI16i16o2i) {
        return "gOhwI16i16o2i";
    }
    if (v == zendnn_gOhwI16i16o4i) {
        return "gOhwI16i16o4i";
    }
    if (v == zendnn_gOdhwI16i16o2i) {
        return "gOdhwI16i16o2i";
    }
    if (v == zendnn_gOdhwI16i16o4i) {
        return "gOdhwI16i16o4i";
    }
    if (v == zendnn_OwI16i32o2i) {
        return "OwI16i32o2i";
    }
    if (v == zendnn_OwI16i32o4i) {
        return "OwI16i32o4i";
    }
    if (v == zendnn_OwI16i48o2i) {
        return "OwI16i48o2i";
    }
    if (v == zendnn_OwI16i48o4i) {
        return "OwI16i48o4i";
    }
    if (v == zendnn_OwI16i64o2i) {
        return "OwI16i64o2i";
    }
    if (v == zendnn_OwI16i64o4i) {
        return "OwI16i64o4i";
    }
    if (v == zendnn_gOwI16i32o2i) {
        return "gOwI16i32o2i";
    }
    if (v == zendnn_gOwI16i32o4i) {
        return "gOwI16i32o4i";
    }
    if (v == zendnn_gOwI16i48o2i) {
        return "gOwI16i48o2i";
    }
    if (v == zendnn_gOwI16i48o4i) {
        return "gOwI16i48o4i";
    }
    if (v == zendnn_gOwI16i64o2i) {
        return "gOwI16i64o2i";
    }
    if (v == zendnn_gOwI16i64o4i) {
        return "gOwI16i64o4i";
    }
    if (v == zendnn_OhwI16i32o2i) {
        return "OhwI16i32o2i";
    }
    if (v == zendnn_OhwI16i32o4i) {
        return "OhwI16i32o4i";
    }
    if (v == zendnn_OhwI16i48o2i) {
        return "OhwI16i48o2i";
    }
    if (v == zendnn_OhwI16i48o4i) {
        return "OhwI16i48o4i";
    }
    if (v == zendnn_OhwI16i64o2i) {
        return "OhwI16i64o2i";
    }
    if (v == zendnn_OhwI16i64o4i) {
        return "OhwI16i64o4i";
    }
    if (v == zendnn_gOhwI16i32o2i) {
        return "gOhwI16i32o2i";
    }
    if (v == zendnn_gOhwI16i32o4i) {
        return "gOhwI16i32o4i";
    }
    if (v == zendnn_gOhwI16i48o2i) {
        return "gOhwI16i48o2i";
    }
    if (v == zendnn_gOhwI16i48o4i) {
        return "gOhwI16i48o4i";
    }
    if (v == zendnn_gOhwI16i64o2i) {
        return "gOhwI16i64o2i";
    }
    if (v == zendnn_gOhwI16i64o4i) {
        return "gOhwI16i64o4i";
    }
    if (v == zendnn_OdhwI16i32o2i) {
        return "OdhwI16i32o2i";
    }
    if (v == zendnn_OdhwI16i32o4i) {
        return "OdhwI16i32o4i";
    }
    if (v == zendnn_OdhwI16i48o2i) {
        return "OdhwI16i48o2i";
    }
    if (v == zendnn_OdhwI16i48o4i) {
        return "OdhwI16i48o4i";
    }
    if (v == zendnn_OdhwI16i64o2i) {
        return "OdhwI16i64o2i";
    }
    if (v == zendnn_OdhwI16i64o4i) {
        return "OdhwI16i64o4i";
    }
    if (v == zendnn_gOdhwI16i32o2i) {
        return "gOdhwI16i32o2i";
    }
    if (v == zendnn_gOdhwI16i32o4i) {
        return "gOdhwI16i32o4i";
    }
    if (v == zendnn_gOdhwI16i48o2i) {
        return "gOdhwI16i48o2i";
    }
    if (v == zendnn_gOdhwI16i48o4i) {
        return "gOdhwI16i48o4i";
    }
    if (v == zendnn_gOdhwI16i64o2i) {
        return "gOdhwI16i64o2i";
    }
    if (v == zendnn_gOdhwI16i64o4i) {
        return "gOdhwI16i64o4i";
    }
    if (v == zendnn_hwioG16g) {
        return "hwioG16g";
    }
    if (v == zendnn_NCdhw40n16c) {
        return "NCdhw40n16c";
    }
    if (v == zendnn_NCw40n16c) {
        return "NCw40n16c";
    }
    if (v == zendnn_NChw40n16c) {
        return "NChw40n16c";
    }
    if (v == zendnn_NCw40n32c) {
        return "NCw40n32c";
    }
    if (v == zendnn_NChw40n32c) {
        return "NChw40n32c";
    }
    if (v == zendnn_NCdhw40n32c) {
        return "NCdhw40n32c";
    }
    if (v == zendnn_OIdhw4o8i8o2i) {
        return "OIdhw4o8i8o2i";
    }
    if (v == zendnn_OIhw4o8i8o2i) {
        return "OIhw4o8i8o2i";
    }
    if (v == zendnn_OIw4o8i8o2i) {
        return "OIw4o8i8o2i";
    }
    if (v == zendnn_gOIdhw4o8i8o2i) {
        return "gOIdhw4o8i8o2i";
    }
    if (v == zendnn_gOIhw4o8i8o2i) {
        return "gOIhw4o8i8o2i";
    }
    if (v == zendnn_gOIw4o8i8o2i) {
        return "gOIw4o8i8o2i";
    }
    if (v == zendnn_IOdhw4i8o8i2o) {
        return "IOdhw4i8o8i2o";
    }
    if (v == zendnn_IOhw4i8o8i2o) {
        return "IOhw4i8o8i2o";
    }
    if (v == zendnn_IOw4i8o8i2o) {
        return "IOw4i8o8i2o";
    }
    if (v == zendnn_gIOdhw4i8o8i2o) {
        return "gIOdhw4i8o8i2o";
    }
    if (v == zendnn_gIOhw4i8o8i2o) {
        return "gIOhw4i8o8i2o";
    }
    if (v == zendnn_gIOw4i8o8i2o) {
        return "gIOw4i8o8i2o";
    }
    if (v == zendnn_NCw2c32n8c) {
        return "NCw2c32n8c";
    }
    if (v == zendnn_NChw2c32n8c) {
        return "NChw2c32n8c";
    }
    if (v == zendnn_NCdhw2c32n8c) {
        return "NCdhw2c32n8c";
    }
    if (v == zendnn_OIw2i8o16i4o) {
        return "OIw2i8o16i4o";
    }
    if (v == zendnn_OIhw2i8o16i4o) {
        return "OIhw2i8o16i4o";
    }
    if (v == zendnn_OIdhw2i8o16i4o) {
        return "OIdhw2i8o16i4o";
    }
    if (v == zendnn_OIw2o8i16o4i) {
        return "OIw2o8i16o4i";
    }
    if (v == zendnn_OIw2o8i16o2i) {
        return "OIw2o8i16o2i";
    }
    if (v == zendnn_IOw2i8o16i4o) {
        return "IOw2i8o16i4o";
    }
    if (v == zendnn_IOw2i8o16i2o) {
        return "IOw2i8o16i2o";
    }
    if (v == zendnn_OIhw2o8i16o4i) {
        return "OIhw2o8i16o4i";
    }
    if (v == zendnn_OIhw2o8i16o2i) {
        return "OIhw2o8i16o2i";
    }
    if (v == zendnn_IOhw2i8o16i4o) {
        return "IOhw2i8o16i4o";
    }
    if (v == zendnn_IOhw2i8o16i2o) {
        return "IOhw2i8o16i2o";
    }
    if (v == zendnn_OIdhw2o8i16o4i) {
        return "OIdhw2o8i16o4i";
    }
    if (v == zendnn_OIdhw2o8i16o2i) {
        return "OIdhw2o8i16o2i";
    }
    if (v == zendnn_IOdhw2i8o16i4o) {
        return "IOdhw2i8o16i4o";
    }
    if (v == zendnn_IOdhw2i8o16i2o) {
        return "IOdhw2i8o16i2o";
    }
    if (v == zendnn_gOIw2o8i16o2i) {
        return "gOIw2o8i16o2i";
    }
    if (v == zendnn_gIOw2i8o16i2o) {
        return "gIOw2i8o16i2o";
    }
    if (v == zendnn_gIOhw2i8o16i2o) {
        return "gIOhw2i8o16i2o";
    }
    if (v == zendnn_gIOdhw2i8o16i2o) {
        return "gIOdhw2i8o16i2o";
    }
    if (v == zendnn_gOIhw2o8i16o2i) {
        return "gOIhw2o8i16o2i";
    }
    if (v == zendnn_gOIdhw2o8i16o2i) {
        return "gOIdhw2o8i16o2i";
    }
    if (v == zendnn_gOIw2o8i16o4i) {
        return "gOIw2o8i16o4i";
    }
    if (v == zendnn_gOIhw2o8i16o4i) {
        return "gOIhw2o8i16o4i";
    }
    assert(!"unknown fmt_tag");
    return "unknown fmt_tag";
}

const char *zendnn_prop_kind2str(zendnn_prop_kind_t v) {
    if (v == zendnn_prop_kind_undef) {
        return "undef";
    }
    if (v == zendnn_forward_training || v == zendnn_forward) {
        return "forward_training or forward";
    }
    if (v == zendnn_forward_inference || v == zendnn_forward_scoring) {
        return "forward_inference or forward_scoring";
    }
    if (v == zendnn_backward) {
        return "backward";
    }
    if (v == zendnn_backward_data) {
        return "backward_data";
    }
    if (v == zendnn_backward_weights) {
        return "backward_weights";
    }
    if (v == zendnn_backward_bias) {
        return "backward_bias";
    }
    assert(!"unknown prop_kind");
    return "unknown prop_kind";
}

const char *zendnn_prim_kind2str(zendnn_primitive_kind_t v) {
    if (v == zendnn_undefined_primitive) {
        return "undef";
    }
    if (v == zendnn_reorder) {
        return "reorder";
    }
    if (v == zendnn_shuffle) {
        return "shuffle";
    }
    if (v == zendnn_concat) {
        return "concat";
    }
    if (v == zendnn_sum) {
        return "sum";
    }
    if (v == zendnn_convolution) {
        return "convolution";
    }
    if (v == zendnn_deconvolution) {
        return "deconvolution";
    }
    if (v == zendnn_eltwise) {
        return "eltwise";
    }
    if (v == zendnn_softmax) {
        return "softmax";
    }
    if (v == zendnn_pooling) {
        return "pooling";
    }
    if (v == zendnn_lrn) {
        return "lrn";
    }
    if (v == zendnn_batch_normalization) {
        return "batch_normalization";
    }
    if (v == zendnn_layer_normalization) {
        return "layer_normalization";
    }
    if (v == zendnn_inner_product) {
        return "inner_product";
    }
    if (v == zendnn_rnn) {
        return "rnn";
    }
    if (v == zendnn_gemm) {
        return "gemm";
    }
    if (v == zendnn_binary) {
        return "binary";
    }
    if (v == zendnn_logsoftmax) {
        return "logsoftmax";
    }
    if (v == zendnn_matmul) {
        return "matmul";
    }
    if (v == zendnn_resampling) {
        return "resampling";
    }
    if (v == zendnn_pooling_v2) {
        return "pooling_v2";
    }
    if (v == zendnn_reduction) {
        return "reduction";
    }
    if (v == zendnn_prelu) {
        return "prelu";
    }
    if (v == zendnn_softmax_v2) {
        return "softmax_v2";
    }
    if (v == zendnn_embedding_bag) {
        return "embedding_bag";
    }
    if (v == zendnn_primitive_kind_max) {
        return "primitive_kind_max";
    }
    assert(!"unknown prim_kind");
    return "unknown prim_kind";
}

const char *zendnn_alg_kind2str(zendnn_alg_kind_t v) {
    if (v == zendnn_alg_kind_undef) {
        return "undef";
    }
    if (v == zendnn_convolution_gemm) {
        return "convolution_gemm";
    }
    if (v == zendnn_convolution_gemm_bf16bf16f32of32) {
        return "convolution_gemm_bf16bf16f32of32";
    }
    if (v == zendnn_convolution_gemm_bf16bf16f32obf16) {
        return "convolution_gemm_bf16bf16f32obf16";
    }
    if (v == zendnn_convolution_gemm_u8s8s16os16) {
        return "convolution_gemm_u8s8s16os16";
    }
    if (v == zendnn_convolution_gemm_u8s8s16os8) {
        return "convolution_gemm_u8s8s16os8";
    }
    if (v == zendnn_convolution_gemm_u8s8s16ou8) {
        return "convolution_gemm_u8s8s16ou8";
    }
    if (v == zendnn_convolution_gemm_u8s8s32os32) {
        return "convolution_gemm_u8s8s32os32";
    }
    if (v == zendnn_convolution_gemm_u8s8s32os8) {
        return "convolution_gemm_u8s8s32os8";
    }
    if (v == zendnn_convolution_gemm_s8s8s32os32) {
        return "convolution_gemm_s8s8s32os32";
    }
    if (v == zendnn_convolution_gemm_s8s8s32os8) {
        return "convolution_gemm_s8s8s32os8";
    }
    if (v == zendnn_convolution_gemm_s8s8s16os16) {
        return "convolution_gemm_s8s8s16os16";
    }
    if (v == zendnn_convolution_gemm_s8s8s16os8) {
        return "convolution_gemm_s8s8s16os8";
    }
    if (v == zendnn_convolution_ref) {
        return "convolution_ref";
    }
    if (v == zendnn_convolution_ck) {
        return "convolution_ck";
    }
    if (v == zendnn_convolution_direct) {
        return "convolution_direct";
    }
    if (v == zendnn_convolution_winograd) {
        return "convolution_winograd";
    }
    if (v == zendnn_convolution_auto) {
        return "convolution_auto";
    }
    if (v == zendnn_deconvolution_direct) {
        return "deconvolution_direct";
    }
    if (v == zendnn_deconvolution_winograd) {
        return "deconvolution_winograd";
    }
    if (v == zendnn_eltwise_relu) {
        return "eltwise_relu";
    }
    if (v == zendnn_eltwise_tanh) {
        return "eltwise_tanh";
    }
    if (v == zendnn_eltwise_elu) {
        return "eltwise_elu";
    }
    if (v == zendnn_eltwise_square) {
        return "eltwise_square";
    }
    if (v == zendnn_eltwise_abs) {
        return "eltwise_abs";
    }
    if (v == zendnn_eltwise_sqrt) {
        return "eltwise_sqrt";
    }
    if (v == zendnn_eltwise_linear) {
        return "eltwise_linear";
    }
    if (v == zendnn_eltwise_bounded_relu) {
        return "eltwise_bounded_relu";
    }
    if (v == zendnn_eltwise_soft_relu) {
        return "eltwise_soft_relu";
    }
    if (v == zendnn_eltwise_logistic) {
        return "eltwise_logistic";
    }
    if (v == zendnn_eltwise_exp) {
        return "eltwise_exp";
    }
    if (v == zendnn_eltwise_gelu_tanh || v == zendnn_eltwise_gelu) {
        return "eltwise_gelu_tanh or eltwise_gelu";
    }
    if (v == zendnn_eltwise_swish) {
        return "eltwise_swish";
    }
    if (v == zendnn_eltwise_log) {
        return "eltwise_log";
    }
    if (v == zendnn_eltwise_clip) {
        return "eltwise_clip";
    }
    if (v == zendnn_eltwise_clip_v2) {
        return "eltwise_clip_v2";
    }
    if (v == zendnn_eltwise_pow) {
        return "eltwise_pow";
    }
    if (v == zendnn_eltwise_gelu_erf) {
        return "eltwise_gelu_erf";
    }
    if (v == zendnn_eltwise_round) {
        return "eltwise_round";
    }
    if (v == zendnn_eltwise_logsigmoid) {
        return "eltwise_logsigmoid";
    }
    if (v == zendnn_eltwise_mish) {
        return "eltwise_mish";
    }
    if (v == zendnn_eltwise_hardswish) {
        return "eltwise_hardswish";
    }
    if (v == zendnn_eltwise_relu_use_dst_for_bwd) {
        return "eltwise_relu_use_dst_for_bwd";
    }
    if (v == zendnn_eltwise_tanh_use_dst_for_bwd) {
        return "eltwise_tanh_use_dst_for_bwd";
    }
    if (v == zendnn_eltwise_elu_use_dst_for_bwd) {
        return "eltwise_elu_use_dst_for_bwd";
    }
    if (v == zendnn_eltwise_sqrt_use_dst_for_bwd) {
        return "eltwise_sqrt_use_dst_for_bwd";
    }
    if (v == zendnn_eltwise_logistic_use_dst_for_bwd) {
        return "eltwise_logistic_use_dst_for_bwd";
    }
    if (v == zendnn_eltwise_exp_use_dst_for_bwd) {
        return "eltwise_exp_use_dst_for_bwd";
    }
    if (v == zendnn_eltwise_clip_v2_use_dst_for_bwd) {
        return "eltwise_clip_v2_use_dst_for_bwd";
    }
    if (v == zendnn_pooling_max) {
        return "pooling_max";
    }
    if (v == zendnn_pooling_avg_include_padding) {
        return "pooling_avg_include_padding";
    }
    if (v == zendnn_pooling_avg_exclude_padding || v == zendnn_pooling_avg) {
        return "pooling_avg_exclude_padding or pooling_avg";
    }
    if (v == zendnn_lrn_across_channels) {
        return "lrn_across_channels";
    }
    if (v == zendnn_lrn_within_channel) {
        return "lrn_within_channel";
    }
    if (v == zendnn_vanilla_rnn) {
        return "vanilla_rnn";
    }
    if (v == zendnn_vanilla_lstm) {
        return "vanilla_lstm";
    }
    if (v == zendnn_vanilla_gru) {
        return "vanilla_gru";
    }
    if (v == zendnn_lbr_gru) {
        return "lbr_gru";
    }
    if (v == zendnn_vanilla_augru) {
        return "vanilla_augru";
    }
    if (v == zendnn_lbr_augru) {
        return "lbr_augru";
    }
    if (v == zendnn_binary_add) {
        return "binary_add";
    }
    if (v == zendnn_binary_mul) {
        return "binary_mul";
    }
    if (v == zendnn_binary_max) {
        return "binary_max";
    }
    if (v == zendnn_binary_min) {
        return "binary_min";
    }
    if (v == zendnn_binary_div) {
        return "binary_div";
    }
    if (v == zendnn_binary_sub) {
        return "binary_sub";
    }
    if (v == zendnn_binary_ge) {
        return "binary_ge";
    }
    if (v == zendnn_binary_gt) {
        return "binary_gt";
    }
    if (v == zendnn_binary_le) {
        return "binary_le";
    }
    if (v == zendnn_binary_lt) {
        return "binary_lt";
    }
    if (v == zendnn_binary_eq) {
        return "binary_eq";
    }
    if (v == zendnn_binary_ne) {
        return "binary_ne";
    }
    if (v == zendnn_resampling_nearest) {
        return "resampling_nearest";
    }
    if (v == zendnn_resampling_linear) {
        return "resampling_linear";
    }
    if (v == zendnn_reduction_max) {
        return "reduction_max";
    }
    if (v == zendnn_reduction_min) {
        return "reduction_min";
    }
    if (v == zendnn_reduction_sum) {
        return "reduction_sum";
    }
    if (v == zendnn_reduction_mul) {
        return "reduction_mul";
    }
    if (v == zendnn_reduction_mean) {
        return "reduction_mean";
    }
    if (v == zendnn_reduction_norm_lp_max) {
        return "reduction_norm_lp_max";
    }
    if (v == zendnn_reduction_norm_lp_sum) {
        return "reduction_norm_lp_sum";
    }
    if (v == zendnn_reduction_norm_lp_power_p_max) {
        return "reduction_norm_lp_power_p_max";
    }
    if (v == zendnn_reduction_norm_lp_power_p_sum) {
        return "reduction_norm_lp_power_p_sum";
    }
    if (v == zendnn_softmax_accurate) {
        return "softmax_accurate";
    }
    if (v == zendnn_softmax_log) {
        return "softmax_log";
    }
    if (v == zendnn_embedding_bag_sum) {
        return "embedding_bag_sum";
    }
    if (v == zendnn_embedding_bag_mean) {
        return "embedding_bag_mean";
    }
    if (v == zendnn_embedding_bag_max) {
        return "embedding_bag_max";
    }
    assert(!"unknown alg_kind");
    return "unknown alg_kind";
}

const char *zendnn_rnn_flags2str(zendnn_rnn_flags_t v) {
    if (v == zendnn_rnn_flags_undef) {
        return "undef";
    }
    assert(!"unknown rnn_flags");
    return "unknown rnn_flags";
}

const char *zendnn_rnn_direction2str(zendnn_rnn_direction_t v) {
    if (v == zendnn_unidirectional_left2right) {
        return "unidirectional_left2right";
    }
    if (v == zendnn_unidirectional_right2left) {
        return "unidirectional_right2left";
    }
    if (v == zendnn_bidirectional_concat) {
        return "bidirectional_concat";
    }
    if (v == zendnn_bidirectional_sum) {
        return "bidirectional_sum";
    }
    //if (v == zendnn_unidirectional) return "unidirectional";
    assert(!"unknown rnn_direction");
    return "unknown rnn_direction";
}

const char *zendnn_engine_kind2str(zendnn_engine_kind_t v) {
    if (v == zendnn_any_engine) {
        return "any";
    }
    if (v == zendnn_cpu) {
        return "cpu";
    }
    if (v == zendnn_gpu) {
        return "gpu";
    }
    assert(!"unknown engine_kind");
    return "unknown engine_kind";
}

const char *zendnn_fpmath_mode2str(zendnn_fpmath_mode_t v) {
    if (v == zendnn_fpmath_mode_strict) {
        return "fpmath_mode_strict";
    }
    if (v == zendnn_fpmath_mode_bf16) {
        return "fpmath_mode_bf16";
    }
    if (v == zendnn_fpmath_mode_f16) {
        return "fpmath_mode_f16";
    }
    if (v == zendnn_fpmath_mode_any) {
        return "any";
    }
    assert(!"unknown fpmath_mode");
    return "unknown fpmath_mode";
}

const char *zendnn_scratchpad_mode2str(zendnn_scratchpad_mode_t v) {
    if (v == zendnn_scratchpad_mode_library) {
        return "library";
    }
    if (v == zendnn_scratchpad_mode_user) {
        return "user";
    }
    assert(!"unknown scratchpad_mode");
    return "unknown scratchpad_mode";
}

const char *zendnn_cpu_isa2str(zendnn_cpu_isa_t v) {
    if (v == zendnn_cpu_isa_all) {
        return "cpu_isa_all";
    }
    if (v == zendnn_cpu_isa_sse41) {
        return "cpu_isa_sse41";
    }
    if (v == zendnn_cpu_isa_avx) {
        return "cpu_isa_avx";
    }
    if (v == zendnn_cpu_isa_avx2) {
        return "cpu_isa_avx2";
    }
    if (v == zendnn_cpu_isa_avx512_mic) {
        return "cpu_isa_avx512_mic";
    }
    if (v == zendnn_cpu_isa_avx512_mic_4ops) {
        return "cpu_isa_avx512_mic_4ops";
    }
    if (v == zendnn_cpu_isa_avx512_core) {
        return "cpu_isa_avx512_core";
    }
    if (v == zendnn_cpu_isa_avx512_core_vnni) {
        return "cpu_isa_avx512_core_vnni";
    }
    if (v == zendnn_cpu_isa_avx512_core_bf16) {
        return "cpu_isa_avx512_core_bf16";
    }
    if (v == zendnn_cpu_isa_avx512_core_amx) {
        return "cpu_isa_avx512_core_amx";
    }
    if (v == zendnn_cpu_isa_avx2_vnni) {
        return "cpu_isa_avx2_vnni";
    }
    assert(!"unknown cpu_isa");
    return "unknown cpu_isa";
}

const char *zendnn_cpu_isa_hints2str(zendnn_cpu_isa_hints_t v) {
    if (v == zendnn_cpu_isa_no_hints) {
        return "cpu_isa_no_hints";
    }
    if (v == zendnn_cpu_isa_prefer_ymm) {
        return "cpu_isa_prefer_ymm";
    }
    assert(!"unknown cpu_isa_hints");
    return "unknown cpu_isa_hints";
}


