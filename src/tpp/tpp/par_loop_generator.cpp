/*****************************************************************************
* Modifications Copyright (c) 2024 Advanced Micro Devices, Inc.
* All rights reserved.
* Notified per clause 4(b) of the license.
******************************************************************************/
/******************************************************************************
* Copyright (c) Intel Corporation - All rights reserved.                      *
* This file is part of the LIBXSMM library.                                   *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/libxsmm/libxsmm/                    *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/

#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <functional>
#include <string>

#ifndef _WIN32
#include <dlfcn.h>
#endif

#define MAX_CODE_SIZE 1048576
//#define STAND_ALONE
namespace zendnn {
namespace tpp {

typedef struct {
  long start;
  long end;
  long step;
  long block_size[8];
} loop_rt_spec_t;

typedef struct {
  int idx_id;
  char idx_name[256];
  char start_var_name[256];
  char end_var_name[256];
  char step_var_name[256];
  int jit_start;
  int jit_step;
  int jit_end;
  int jit_block_sizes;
  long start;
  long end;
  long step;
  int pos_in_loopnest;
  int is_parallelizable;
  int is_blocked;
  int is_blocked_outer;
  long block_size[8];
  int is_par_across_col_teams;
  int is_par_across_row_teams;
  int n_col_teams;
  int n_row_teams;
} loop_param_t;

typedef struct {
  char* buf;
  int cur_nest_level;
  int cur_pos;
  int n_loops;
  loop_param_t* loop_params;
  int n_logical_loops;
  char occurence_map[256];
  int jit_loop_spec;
  int use_2d_par;
  int n_row_teams;
  int n_col_teams;
} loop_code;

loop_param_t find_loop_param_at_pos(loop_param_t* i_loop_params, int pos) {
  loop_param_t res;
  int i = 0;
  int found = 0;
  while (!found) {
    if (i_loop_params[i].pos_in_loopnest == pos) {
      found = 1;
      res = i_loop_params[i];
    } else {
      i++;
    }
  }
  return res;
}

void add_buf_to_code(loop_code* i_code, char* buf) {
  snprintf(
      i_code->buf + i_code->cur_pos,
      (MAX_CODE_SIZE - i_code->cur_pos) * sizeof(char),
      "%s",
      buf);
  i_code->cur_pos += strlen(buf);
}

void align_line(loop_code* i_code) {
  char tmp_buf[512];
  int i;
  for (i = 0; i < 2 * i_code->cur_nest_level; i++) {
    tmp_buf[i] = ' ';
  }
  tmp_buf[2 * i_code->cur_nest_level] = '\0';
  add_buf_to_code(i_code, tmp_buf);
  return;
}

void increase_nest_level(loop_code* i_code) {
  i_code->cur_nest_level = i_code->cur_nest_level + 1;
  return;
}

void decrease_nest_level(loop_code* i_code) {
  i_code->cur_nest_level = i_code->cur_nest_level - 1;
  return;
}

void emit_parallel_for(loop_code* i_code, int collapse_level) {
  char tmp_buf[512];
  align_line(i_code);
  if (collapse_level > 1) {
    snprintf(
        tmp_buf,
        sizeof(tmp_buf),
        "#pragma omp for collapse(%d) nowait\n",
        collapse_level);
  } else {
    snprintf(tmp_buf, sizeof(tmp_buf), "#pragma omp for nowait\n");
  }
  add_buf_to_code(i_code, tmp_buf);
  return;
}

void emit_loop_header(loop_code* i_code) {
  char tmp_buf[512];
  align_line(i_code);
  snprintf(tmp_buf, sizeof(tmp_buf), "#pragma omp parallel\n");
}

void emit_parallel_region(loop_code* i_code) {
  char tmp_buf[512];
  align_line(i_code);
  snprintf(tmp_buf, sizeof(tmp_buf), "#pragma omp parallel\n");
  add_buf_to_code(i_code, tmp_buf);
  align_line(i_code);
  snprintf(tmp_buf, sizeof(tmp_buf), "{\n");
  add_buf_to_code(i_code, tmp_buf);
  increase_nest_level(i_code);
  if (i_code->use_2d_par > 0) {
    align_line(i_code);
    snprintf(tmp_buf, sizeof(tmp_buf), "int tid = omp_get_thread_num();\n");
    add_buf_to_code(i_code, tmp_buf);
    align_line(i_code);
    snprintf(
        tmp_buf, sizeof(tmp_buf), "int row_teams = %d;\n", i_code->n_row_teams);
    add_buf_to_code(i_code, tmp_buf);
    align_line(i_code);
    snprintf(
        tmp_buf, sizeof(tmp_buf), "int col_teams = %d;\n", i_code->n_col_teams);
    add_buf_to_code(i_code, tmp_buf);
    align_line(i_code);
    snprintf(tmp_buf, sizeof(tmp_buf), "int row_id = tid/col_teams;\n");
    add_buf_to_code(i_code, tmp_buf);
    align_line(i_code);
    snprintf(tmp_buf, sizeof(tmp_buf), "int col_id = tid%%col_teams;\n");
    add_buf_to_code(i_code, tmp_buf);
    align_line(i_code);
    snprintf(tmp_buf, sizeof(tmp_buf), "if (tid < row_teams * col_teams) {\n");
    add_buf_to_code(i_code, tmp_buf);
    increase_nest_level(i_code);
  }
  return;
}

void close_parallel_region(loop_code* i_code) {
  char tmp_buf[512];
  decrease_nest_level(i_code);
  align_line(i_code);
  snprintf(tmp_buf, sizeof(tmp_buf), "}\n");
  add_buf_to_code(i_code, tmp_buf);
  return;
}

void emit_loop_header(loop_code* i_code, loop_param_t* i_loop_param) {
  char tmp_buf[512];
  char str_idx[512];
  char str_start[512];
  char str_end[512];
  char str_step[512];

  if (strcmp(i_loop_param->idx_name, "") == 0) {
    snprintf(str_idx, sizeof(str_idx), "i%d", i_loop_param->idx_id);
  } else {
    snprintf(str_idx, sizeof(str_idx), "%s", i_loop_param->idx_name);
  }

  if (strcmp(i_loop_param->start_var_name, "") == 0) {
    snprintf(str_start, sizeof(str_start), "%ld", i_loop_param->start);
  } else {
    snprintf(str_start, sizeof(str_start), "%s", i_loop_param->start_var_name);
  }

  if (strcmp(i_loop_param->end_var_name, "") == 0) {
    snprintf(str_end, sizeof(str_end), "%ld", i_loop_param->end);
  } else {
    snprintf(str_end, sizeof(str_end), "%s", i_loop_param->end_var_name);
  }

  if (strcmp(i_loop_param->step_var_name, "") == 0) {
    snprintf(str_step, sizeof(str_step), "%ld", i_loop_param->step);
  } else {
    snprintf(str_step, sizeof(str_step), "%s", i_loop_param->step_var_name);
  }

  if ((i_loop_param->is_par_across_col_teams > 0) ||
      (i_loop_param->is_par_across_row_teams > 0)) {
    char prefix[16];
    if (i_loop_param->is_par_across_col_teams > 0) {
      snprintf(prefix, sizeof(prefix), "col");
    } else {
      snprintf(prefix, sizeof(prefix), "row");
    }
    align_line(i_code);
    snprintf(
        tmp_buf,
        sizeof(tmp_buf),
        "int %s_tasks = ((%s) - (%s) + ((%s) - 1))/(%s);\n",
        prefix,
        str_end,
        str_start,
        str_step,
        str_step);
    add_buf_to_code(i_code, tmp_buf);
    align_line(i_code);
    snprintf(
        tmp_buf,
        sizeof(tmp_buf),
        "int %s_tasks_chunksize = (%s_tasks + %s_teams - 1)/%s_teams;\n",
        prefix,
        prefix,
        prefix,
        prefix);
    add_buf_to_code(i_code, tmp_buf);
    align_line(i_code);
    snprintf(
        tmp_buf,
        sizeof(tmp_buf),
        "int my_%s_start = (%s_id * %s_tasks_chunksize < %s_tasks) ? %s + (%s_id * %s_tasks_chunksize) * %s : %s;\n",
        prefix,
        prefix,
        prefix,
        prefix,
        str_start,
        prefix,
        prefix,
        str_step,
        str_end);
    add_buf_to_code(i_code, tmp_buf);
    align_line(i_code);
    snprintf(
        tmp_buf,
        sizeof(tmp_buf),
        "int my_%s_end = ((%s_id+1) * %s_tasks_chunksize < %s_tasks) ? %s + ((%s_id+1) * %s_tasks_chunksize) * %s : %s;\n",
        prefix,
        prefix,
        prefix,
        prefix,
        str_start,
        prefix,
        prefix,
        str_step,
        str_end);
    add_buf_to_code(i_code, tmp_buf);
    align_line(i_code);
    snprintf(
        tmp_buf,
        sizeof(tmp_buf),
        "for (int %s = my_%s_start; %s < my_%s_end; %s += %s) {\n",
        str_idx,
        prefix,
        str_idx,
        prefix,
        str_idx,
        str_step);
    add_buf_to_code(i_code, tmp_buf);
    increase_nest_level(i_code);
  } else {
    align_line(i_code);
    snprintf(
        tmp_buf,
        sizeof(tmp_buf),
        "for (int %s = %s; %s < %s; %s += %s) {\n",
        str_idx,
        str_start,
        str_idx,
        str_end,
        str_idx,
        str_step);
    add_buf_to_code(i_code, tmp_buf);
    increase_nest_level(i_code);
  }

  return;
}

void emit_func_signature(
    loop_code* i_code,
    char* spec_func_name,
    char* body_func_name,
    char* init_func_name,
    char* term_func_name) {
  char tmp_buf[512];
  // int i;
  align_line(i_code);
  snprintf(
      tmp_buf,
      sizeof(tmp_buf),
      "#include <omp.h>\nextern \"C\" void par_nested_loops(loop_rt_spec_t *%s, std::function<void(int *)> %s, std::function<void()> %s, std::function<void()> %s) {\n",
      spec_func_name,
      body_func_name,
      init_func_name,
      term_func_name);
  add_buf_to_code(i_code, tmp_buf);
  increase_nest_level(i_code);
}

void emit_func_termination(loop_code* i_code) {
  char tmp_buf[512];
  decrease_nest_level(i_code);
  align_line(i_code);
  snprintf(tmp_buf, sizeof(tmp_buf), "}\n");
  add_buf_to_code(i_code, tmp_buf);
  if (i_code->use_2d_par > 0) {
    decrease_nest_level(i_code);
    align_line(i_code);
    snprintf(tmp_buf, sizeof(tmp_buf), "}\n");
    add_buf_to_code(i_code, tmp_buf);
  }
  return;
}

void emit_void_function(loop_code* i_code, char* func_name) {
  char tmp_buf[512];
  align_line(i_code);
  snprintf(tmp_buf, sizeof(tmp_buf), "if (%s) %s();\n", func_name, func_name);
  add_buf_to_code(i_code, tmp_buf);
  return;
}

void emit_loop_body(loop_code* i_code, char* body_func_name) {
  char tmp_buf[512];
  int i;
  align_line(i_code);
  snprintf(tmp_buf, sizeof(tmp_buf), "int idx[%d];\n", i_code->n_logical_loops);
  add_buf_to_code(i_code, tmp_buf);
  /* Here we set the idx array to be used by function called */
  for (i = 0; i < i_code->n_logical_loops; i++) {
    char str_idx[64];
    snprintf(
        str_idx,
        sizeof(tmp_buf),
        "%c%d",
        'a' + i,
        i_code->occurence_map['a' + i] - 1);
    align_line(i_code);
    snprintf(tmp_buf, sizeof(tmp_buf), "idx[%d] = %s;\n", i, str_idx);
    add_buf_to_code(i_code, tmp_buf);
  }
  align_line(i_code);
  snprintf(tmp_buf, sizeof(tmp_buf), "%s(idx);\n", body_func_name);
  add_buf_to_code(i_code, tmp_buf);
  return;
}

void emit_loop_termination(loop_code* i_code) {
  char tmp_buf[512];
  decrease_nest_level(i_code);
  align_line(i_code);
  snprintf(tmp_buf, sizeof(tmp_buf), "}\n");
  add_buf_to_code(i_code, tmp_buf);
  return;
}

void emit_barrier(loop_code* i_code) {
  char tmp_buf[512];
  align_line(i_code);
  snprintf(tmp_buf, sizeof(tmp_buf), "#pragma omp barrier\n");
  add_buf_to_code(i_code, tmp_buf);
  return;
}

void set_loop_param(
    loop_param_t* io_param,
    const char* idx_name,
    const char* s_name,
    const char* e_name,
    const char* step_name,
    int pos) {
  io_param->pos_in_loopnest = pos;
  snprintf(io_param->idx_name, sizeof(io_param->idx_name), "%s", idx_name);
  snprintf(
      io_param->start_var_name, sizeof(io_param->start_var_name), "%s", s_name);
  snprintf(
      io_param->end_var_name, sizeof(io_param->end_var_name), "%s", e_name);
  snprintf(
      io_param->step_var_name,
      sizeof(io_param->step_var_name),
      "%s",
      step_name);
  return;
}

int is_simple_char(char cur) {
  int result = 0;
  if ((cur >= 'a' && cur <= 'z') || (cur >= 'A' && cur <= 'Z') ||
      (cur == '|')) {
    result = 1;
  }
  return result;
}

void parse_jit_info(char* jit_info_str, loop_param_t* loop_param) {
  char cur_token[512];
  char token_start[512];
  char token_end[512];
  char token_step[512];
  char token_bs[512];
  int i = 0;
  int j = 0;
  int token_id = 0;
  char* bs_str;
  int bs_index = 0;

  /* First extract the BS token */
  while (jit_info_str[i] != '(') {
    i++;
  }
  jit_info_str[i] = '\0';
  i++;
  while (jit_info_str[i] != ')') {
    token_bs[j] = jit_info_str[i];
    j++;
    i++;
  }
  token_bs[j] = '\0';

  /* Now extract rest token */
  i = 0;
  j = 0;
  while (jit_info_str[i] != '\0') {
    if (jit_info_str[i] == ',') {
      if (i == 0) {
        /* Empty token */
        if (token_id == 0) {
          snprintf(token_start, sizeof(token_start), "");
        } else if (token_id == 1) {
          snprintf(token_end, sizeof(token_end), "");
        } else if (token_id == 2) {
          snprintf(token_step, sizeof(token_step), "");
        }
        token_id++;
      } else if (jit_info_str[i - 1] == ',') {
        /* Empty token */
        if (token_id == 0) {
          snprintf(token_start, sizeof(token_start), "");
        } else if (token_id == 1) {
          snprintf(token_end, sizeof(token_end), "");
        } else if (token_id == 2) {
          snprintf(token_step, sizeof(token_step), "");
        }
        token_id++;
      } else {
        /* Finalize current token */
        cur_token[j] = '\0';
        j = 0;
        if (token_id == 0) {
          snprintf(token_start, sizeof(token_start), "%s", cur_token);
        } else if (token_id == 1) {
          snprintf(token_end, sizeof(token_end), "%s", cur_token);
        } else if (token_id == 2) {
          snprintf(token_step, sizeof(token_step), "%s", cur_token);
        }
        token_id++;
      }
    } else {
      cur_token[j] = jit_info_str[i];
      j++;
    }
    i++;
  }

  /* Now based on token parse info... */
  if (strlen(token_start) > 0) {
    loop_param->jit_start = 1;
    loop_param->start = atoi(token_start);
  }

  if (strlen(token_end) > 0) {
    loop_param->jit_end = 1;
    loop_param->end = atoi(token_end);
  }

  if (strlen(token_step) > 0) {
    loop_param->jit_step = 1;
    loop_param->step = atoi(token_step);
  }

  if (strlen(token_bs) > 0) {
    bs_str = strtok(token_bs, ",");
    bs_index = 0;
    while (bs_str != NULL) {
      loop_param->jit_block_sizes = 1;
      loop_param->block_size[bs_index] = atoi(bs_str);
      bs_index++;
      bs_str = strtok(NULL, ",");
    }
  }
}

void extract_jit_info(
    const char* in_desc,
    char* out_desc,
    loop_param_t* loop_params) {
  int i = 0, k = 0;
  char jit_params_str[512];
  char loop_id;

  while (i < strlen(in_desc)) {
    char cur = in_desc[i];
    if (is_simple_char(cur)) {
      out_desc[k] = cur;
      k++;
      i++;
      if (cur != '|') {
        loop_id = tolower(cur);
      }
    } else {
      /* Start reading specs string [ .. ] */
      if (cur == '[') {
        int j = 0;
        while (cur != ']') {
          i++;
          cur = in_desc[i];
          if (cur != ']') {
            jit_params_str[j] = cur;
            j++;
          } else {
            i++;
          }
        }
        jit_params_str[j] = '\0';
        parse_jit_info(jit_params_str, &loop_params[loop_id - 'a']);
      }
    }
  }
  out_desc[k] = '\0';
}

void extract_2d_par_info(
    const char* in_desc,
    char* out_desc,
    loop_param_t* loop_params,
    loop_code* i_code) {
  int i = 0, k = 0;
  char jit_params_str[512];
  char loop_id = 0;

  while (i < strlen(in_desc)) {
    char cur = in_desc[i];
    if (cur != '{') {
      out_desc[k] = cur;
      k++;
      i++;
      if (is_simple_char(cur) && cur != '|') {
        loop_id++;
      }
    } else {
      /* Start reading parallelization string {R or C:parallelization degree}]
       */
      int j = 0;
      i++;
      /* Consume par dimension */
      cur = in_desc[i];
      if (cur == 'R' || cur == 'r') {
        loop_params[loop_id - 1].is_par_across_row_teams = 1;
        loop_params[loop_id - 1].is_par_across_col_teams = 0;
      } else if (cur == 'C' || cur == 'c') {
        loop_params[loop_id - 1].is_par_across_col_teams = 1;
        loop_params[loop_id - 1].is_par_across_row_teams = 0;
      }
      /* Consume :  */
      i += 2;
      cur = in_desc[i];
      while (cur != '}') {
        jit_params_str[j] = cur;
        j++;
        i++;
        cur = in_desc[i];
      }
      i++;
      jit_params_str[j] = '\0';
      if (loop_params[loop_id - 1].is_par_across_row_teams == 1) {
        loop_params[loop_id - 1].n_row_teams = atoi(jit_params_str);
        i_code->n_row_teams = loop_params[loop_id - 1].n_row_teams;
      } else {
        loop_params[loop_id - 1].n_col_teams = atoi(jit_params_str);
        i_code->n_col_teams = loop_params[loop_id - 1].n_col_teams;
      }
    }
  }
  out_desc[k] = '\0';
}

// void loop_generator( FILE *fp_out, const char *__loop_nest_desc_extended ) {
std::string loop_generator(const char* __loop_nest_desc_extended) {
  char body_func_name[64] = "body_func";
  char init_func_name[64] = "init_func";
  char term_func_name[64] = "term_func";
  char spec_func_name[64] = "loop_rt_spec";
  char loop_map[256];
  char occurence_map[256];
  loop_code l_code;
  char* result_code;
  loop_param_t loop_params[256], cur_loop, loop_params_map[256];
  int n_loops, n_logical_loops, i, k, have_emitted_parallel_for = 0,
                                      n_parallel_loops = 0;
  char loop_nest_desc[256];
  char barrier_positions[256];
  int jit_loop_spec = 0;
  int use_2d_par = 0;
  size_t src_len = strlen(__loop_nest_desc_extended);
  char _loop_nest_desc_extended[src_len];
  char loop_nest_desc_extended[src_len];

  /* Extract explicit 2D parallelization info */
  for (i = 0; i < src_len; i++) {
    if (__loop_nest_desc_extended[i] == '{') {
      use_2d_par = 1;
      break;
    }
  }
  l_code.use_2d_par = use_2d_par;
  if (use_2d_par > 0) {
    l_code.n_col_teams = 1;
    l_code.n_row_teams = 1;
    extract_2d_par_info(
        __loop_nest_desc_extended,
        _loop_nest_desc_extended,
        loop_params,
        &l_code);
  } else {
    strncpy(_loop_nest_desc_extended, __loop_nest_desc_extended, src_len);
    _loop_nest_desc_extended[src_len] = '\0';
  }

  /* Check if we have to jit the loop specs  */
  for (i = 0; i < src_len; i++) {
    if (_loop_nest_desc_extended[i] == '[') {
      jit_loop_spec = 1;
      break;
    }
  }
  l_code.jit_loop_spec = jit_loop_spec;

  std::fill_n(loop_params_map, 256, loop_param_t{});
  if (jit_loop_spec > 0) {
    extract_jit_info(
        _loop_nest_desc_extended, loop_nest_desc_extended, loop_params_map);
  } else {
    strncpy(loop_nest_desc_extended, _loop_nest_desc_extended, src_len);
    loop_nest_desc_extended[src_len] = '\0';
  }

  /* Cleanup input descriptor to exclude barriers */
  k = 0;
  std::fill_n(barrier_positions, 256, 0);
  for (i = 0; i < src_len; i++) {
    if (loop_nest_desc_extended[i] == '|') {
      if (k - 1 >= 0) {
        barrier_positions[k - 1] = 1;
      }
    } else {
      loop_nest_desc[k] = loop_nest_desc_extended[i];
      k++;
    }
  }
  loop_nest_desc[k] = '\0';

  n_loops = strlen(loop_nest_desc);
  result_code = (char*)malloc(MAX_CODE_SIZE * sizeof(char));

  l_code.buf = result_code;
  l_code.cur_nest_level = 0;
  l_code.n_loops = n_loops;
  l_code.loop_params = loop_params;
  l_code.cur_pos = 0;

  /* Find number of parallel loops */
  for (i = 0; i < n_loops; i++) {
    if (tolower(loop_nest_desc[i]) != loop_nest_desc[i]) {
      n_parallel_loops++;
    }
  }

  /* Count how many times each loop occurs (lower case and upper case are
   * equivalent for that matter) */
  std::fill_n(loop_map, 256, 0);
  for (i = 0; i < n_loops; i++) {
    loop_map[tolower(loop_nest_desc[i])]++;
  }

  /* Set up loop properties */
  std::fill_n(occurence_map, 256, 0);
  for (i = 0; i < n_loops; i++) {
    int is_blocked = (loop_map[tolower(loop_nest_desc[i])] > 1) ? 1 : 0;
    int is_parallelizable =
        (tolower(loop_nest_desc[i]) != loop_nest_desc[i]) ? 1 : 0;
    int occurence_id, is_blocked_outer;
    char idx_name[16];
    char spec_array_name[512];
    char start_var_name[512];
    char end_var_name[512];
    char step_var_name[512];
    int loop_abs_index = tolower(loop_nest_desc[i]) - 'a';

    occurence_id = occurence_map[tolower(loop_nest_desc[i])];
    is_blocked_outer = (occurence_id == 0) ? 1 : 0;
    occurence_map[tolower(loop_nest_desc[i])]++;

    snprintf(spec_array_name, sizeof(spec_array_name), "%s", spec_func_name);

    snprintf(
        idx_name,
        sizeof(idx_name),
        "%c%d",
        tolower(loop_nest_desc[i]),
        occurence_id);

    if (occurence_id == 0) {
      if (loop_params_map[loop_abs_index].jit_start > 0) {
        snprintf(
            start_var_name,
            sizeof(start_var_name),
            "%ld",
            loop_params_map[loop_abs_index].start);
      } else {
        snprintf(
            start_var_name,
            sizeof(start_var_name),
            "%s[%d].start",
            spec_array_name,
            loop_abs_index);
      }
    } else {
      snprintf(
          start_var_name,
          sizeof(start_var_name),
          "%c%d",
          tolower(loop_nest_desc[i]),
          occurence_id - 1);
    }

    if (occurence_id == 0) {
      if (loop_params_map[loop_abs_index].jit_end > 0) {
        snprintf(
            end_var_name,
            sizeof(end_var_name),
            "%ld",
            loop_params_map[loop_abs_index].end);
      } else {
        snprintf(
            end_var_name,
            sizeof(end_var_name),
            "%s[%d].end",
            spec_array_name,
            loop_abs_index);
      }
    } else {
      if (loop_params_map[loop_abs_index].jit_block_sizes > 0) {
        snprintf(
            end_var_name,
            sizeof(end_var_name),
            "%c%d + %ld",
            tolower(loop_nest_desc[i]),
            occurence_id - 1,
            loop_params_map[loop_abs_index].block_size[occurence_id - 1]);
      } else {
        snprintf(
            end_var_name,
            sizeof(end_var_name),
            "%c%d + %s[%d].block_size[%d]",
            tolower(loop_nest_desc[i]),
            occurence_id - 1,
            spec_array_name,
            loop_abs_index,
            occurence_id - 1);
      }
    }

    if (is_blocked) {
      if (occurence_id == loop_map[tolower(loop_nest_desc[i])] - 1) {
        if (loop_params_map[loop_abs_index].jit_step > 0) {
          snprintf(
              step_var_name,
              sizeof(step_var_name),
              "%ld",
              loop_params_map[loop_abs_index].step);
        } else {
          snprintf(
              step_var_name,
              sizeof(step_var_name),
              "%s[%d].step",
              spec_array_name,
              loop_abs_index);
        }
      } else {
        if (loop_params_map[loop_abs_index].jit_block_sizes > 0) {
          snprintf(
              step_var_name,
              sizeof(step_var_name),
              "%ld",
              loop_params_map[loop_abs_index].block_size[occurence_id]);
        } else {
          snprintf(
              step_var_name,
              sizeof(step_var_name),
              "%s[%d].block_size[%d]",
              spec_array_name,
              loop_abs_index,
              occurence_id);
        }
      }
    } else {
      if (loop_params_map[loop_abs_index].jit_step > 0) {
        snprintf(
            step_var_name,
            sizeof(step_var_name),
            "%ld",
            loop_params_map[loop_abs_index].step);
      } else {
        snprintf(
            step_var_name,
            sizeof(step_var_name),
            "%s[%d].step",
            spec_array_name,
            loop_abs_index);
      }
    }

    set_loop_param(
        &loop_params[i],
        idx_name,
        start_var_name,
        end_var_name,
        step_var_name,
        i);
    loop_params[i].is_parallelizable = is_parallelizable;
    loop_params[i].is_blocked = is_blocked;
    loop_params[i].is_blocked_outer = is_blocked_outer;
  }

  /* Setup number of logical loops and the ocurence map */
  n_logical_loops = 0;
  for (i = 0; i < 256; i++) {
    if (occurence_map[i] > 0) {
      n_logical_loops++;
    }
  }
  l_code.n_logical_loops = n_logical_loops;

  memcpy(&l_code.occurence_map[0], occurence_map, 256);

  /* Emit function signature  */
  emit_func_signature(
      &l_code, spec_func_name, body_func_name, init_func_name, term_func_name);

  /* Emit loop function header */
  if (n_parallel_loops > 0) {
    emit_parallel_region(&l_code);
  }

  /* Emit init function */
  emit_void_function(&l_code, init_func_name);

  for (i = 0; i < n_loops; i++) {
    cur_loop = loop_params[i];
    /* Emit parallel for if need be*/
    if ((cur_loop.is_parallelizable == 1) && (have_emitted_parallel_for == 0) &&
        (cur_loop.is_par_across_col_teams == 0) &&
        (cur_loop.is_par_across_row_teams == 0)) {
      int collapse_level = 1;
      int j = i + 1;
      int is_parallel = 1;
      while ((is_parallel > 0) && (j < n_loops)) {
        loop_param_t tmp_loop = loop_params[j];
        if (tmp_loop.is_parallelizable > 0) {
          collapse_level++;
          j++;
        } else {
          is_parallel = 0;
        }
      }
      emit_parallel_for(&l_code, collapse_level);
      have_emitted_parallel_for = 1;
    }
    emit_loop_header(&l_code, &cur_loop);
  }

  emit_loop_body(&l_code, body_func_name);

  for (i = n_loops - 1; i >= 0; i--) {
    emit_loop_termination(&l_code);
    if (barrier_positions[i] > 0) {
      emit_barrier(&l_code);
    }
  }

  /* Emit term function */
  emit_void_function(&l_code, term_func_name);

  if (n_parallel_loops > 0) {
    close_parallel_region(&l_code);
  }

  emit_func_termination(&l_code);

  // fprintf(fp_out, "%s", result_code);
  // fprintf(stderr, "%s", result_code);
  std::string outstr = std::string(result_code);

  if (result_code)
    free(result_code);

  return outstr;
}

} // namespace tpp
} // namespace zendnn
