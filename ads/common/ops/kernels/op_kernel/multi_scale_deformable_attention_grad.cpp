/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2023. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/*!
 * \file multi_scale_deformable_attention_grad.h
 * \brief
 */
#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
using namespace AscendC;

namespace
{
    constexpr static int32_t INPUT_NUM = 6;
    constexpr static int32_t OUTPUT_NUM = 3;
    constexpr static int32_t VALUE_INPUT_INDEX = 0;
    constexpr static int32_t SS_INPUT_INDEX = 1;
    constexpr static int32_t LSI_INPUT_INDEX = 2;
    constexpr static int32_t SL_INPUT_INDEX = 3;
    constexpr static int32_t AW_INPUT_INDEX = 4;
    constexpr static int32_t GO_INPUT_INDEX = 5;
    constexpr static int32_t GV_OUTPUT_INDEX = 6;
    constexpr static int32_t GSL_OUTPUT_INDEX = 7;
    constexpr static int32_t GAW_OUTPUT_INDEX = 8;
    constexpr static int32_t BUFFER_NUM = 2;
    constexpr static int32_t DOUB = 2;
    constexpr static int32_t T_BLOCK = 8;
    constexpr static uint16_t DST_BLK_STRIDE = 1;
    constexpr static uint16_t SRC_BLK_STRIDE = 1;
    constexpr static uint8_t DST_REP_STRIDE = 8;
    constexpr static uint8_t SRC_REP_STRIDE = 8;
};

template <typename T>
class MultiScaleDeformableAttentionGrad
{
public:
    __aicore__ inline MultiScaleDeformableAttentionGrad(){};
    __aicore__ inline void init(GM_ADDR input_tensors[INPUT_NUM + OUTPUT_NUM], MultiScaleDeformableAttentionGradTilingData *tiling_data);
    __aicore__ inline void init_buffer();
    __aicore__ inline void init_local_tensor();
    __aicore__ inline void process_grad_value_with_point(int32_t cur_nh, int32_t cur_nl,
                                                         int32_t base_ptr, int32_t value_ptr_offset,
                                                         int32_t h, int32_t w);
    __aicore__ inline void process_grad_weight_with_point(int32_t cur_nh, int32_t cur_np);
    __aicore__ inline void process();
    __aicore__ inline void compute_mode_zero();
    __aicore__ inline int32_t ceil(int32_t a, int32_t b);
    __aicore__ inline void muls_template(const LocalTensor<T> &dstLocal,
                                         const LocalTensor<T> &srcLocal,
                                         T scalarValue, const int32_t calCount);
    __aicore__ inline void muls_template_int32(const LocalTensor<int32_t> &dstLocal,
                                               const LocalTensor<int32_t> &srcLocal,
                                               int32_t scalarValue, const int32_t calCount);
    __aicore__ inline void adds_template(const LocalTensor<T> &dstLocal,
                                         const LocalTensor<T> &srcLocal,
                                         T scalarValue, const int32_t calCount);
    __aicore__ inline void adds_template_int32(const LocalTensor<int32_t> &dstLocal,
                                               const LocalTensor<int32_t> &srcLocal,
                                               int32_t scalarValue, const int32_t calCount);
    __aicore__ inline void process_levels(int32_t cur_nh, int32_t base_ptr,
                                          int32_t cur_b, int32_t cur_q, int32_t sl_size,
                                          int32_t qid_stride, int32_t data_value_ptr_init_offset,
                                          int32_t w_stride, int32_t sample_location_offset,
                                          int32_t data_weight_ptr, int32_t data_loc_w_ptr);
    __aicore__ inline void pre_process_levels(int32_t cur_nh, int32_t cur_b,
                                              int32_t cur_nl, int32_t cur_q,
                                              int32_t sl_size,int32_t h_stride,
                                              int32_t w_stride, int32_t loc_h_offset,
                                              int32_t loc_w_offset, int32_t w, int32_t h,
                                              int32_t sample_location_offset);
    __aicore__ inline void post_process_levels(int32_t w, int32_t h, int32_t cur_nl, int32_t data_weight_ptr,
                                               int32_t data_loc_w_ptr, int32_t cur_nh);
    __aicore__ inline void cal_grad_value(LocalTensor<T> &v_ub, LocalTensor<int32_t> &offset1_ub,
                                          LocalTensor<int32_t> &offset2_ub, LocalTensor<T> &h_w_w1_ub,
                                          LocalTensor<T> &h_w_w2_ub, LocalTensor<T> &w_weight_ub,
                                          int32_t cur_np, int32_t base_ptr, int32_t value_ptr_offset,
                                          bool neg_h, bool neg_w);
private:
    TPipe pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM> in_queue_grad_output, in_queue_lsi, in_queue_sl, in_queue_aw, in_queue_ss;
    GlobalTensor<T> value_gm, sampling_loc_gm, attn_weight_gm, grad_output_gm;
    GlobalTensor<int32_t> level_start_index_gm, spatial_shapes_gm;
    GlobalTensor<T> grad_value_gm, grad_sampling_loc_gm, grad_attn_weight_gm;
    TBuf<TPosition::VECCALC> buffer_h_im;
    TBuf<TPosition::VECCALC> buffer_w_im;
    TBuf<TPosition::VECCALC> buffer_h_low;
    TBuf<TPosition::VECCALC> buffer_w_low;
    TBuf<TPosition::VECCALC> buffer_h_high;
    TBuf<TPosition::VECCALC> buffer_w_high;
    TBuf<TPosition::VECCALC> buffer_h_low_t;
    TBuf<TPosition::VECCALC> buffer_w_low_t;
    TBuf<TPosition::VECCALC> buffer_lh;
    TBuf<TPosition::VECCALC> buffer_lw;
    TBuf<TPosition::VECCALC> buffer_neg_lh;
    TBuf<TPosition::VECCALC> buffer_neg_lw;
    TBuf<TPosition::VECCALC> buffer_hh;
    TBuf<TPosition::VECCALC> buffer_hw;
    TBuf<TPosition::VECCALC> buffer_h_low_ptr_offset;
    TBuf<TPosition::VECCALC> buffer_h_high_ptr_offset;
    TBuf<TPosition::VECCALC> buffer_w_low_ptr_offset;
    TBuf<TPosition::VECCALC> buffer_w_high_ptr_offset;
    TBuf<TPosition::VECCALC> buffer_w1;
    TBuf<TPosition::VECCALC> buffer_w2;
    TBuf<TPosition::VECCALC> buffer_w3;
    TBuf<TPosition::VECCALC> buffer_w4;
    TBuf<TPosition::VECCALC> buffer_grad_h_weight;
    TBuf<TPosition::VECCALC> buffer_grad_w_weight;
    TBuf<TPosition::VECCALC> buffer_top_grad_value;
    TBuf<TPosition::VECCALC> buffer_v1;
    TBuf<TPosition::VECCALC> buffer_v2;
    TBuf<TPosition::VECCALC> buffer_v3;
    TBuf<TPosition::VECCALC> buffer_v4;
    TBuf<TPosition::VECCALC> buffer_v_w1;
    TBuf<TPosition::VECCALC> buffer_v_w2;
    TBuf<TPosition::VECCALC> buffer_mid;
    TBuf<TPosition::VECCALC> buffer_w1_v1;
    TBuf<TPosition::VECCALC> buffer_w2_v2;
    TBuf<TPosition::VECCALC> buffer_w3_v3;
    TBuf<TPosition::VECCALC> buffer_w4_v4;
    TBuf<TPosition::VECCALC> buffer_val;
    TBuf<TPosition::VECCALC> buffer_grad_weight;
    TBuf<TPosition::VECCALC> buffer_grad_weight_full;
    TBuf<TPosition::VECCALC> buffer_grad_sample_loc;
    int32_t spatial_size;
    int32_t cur_block_idx;
    int32_t cur_core_task_num;
    int32_t num_heads;
    int32_t channels;
    int32_t t_per_block;
    int32_t int32_per_block;
    int32_t per_ub_size;
    int32_t start_task_id;
    int32_t num_levels;
    int32_t num_query;
    int32_t num_point;
    int32_t num_point_align;
    int32_t channel_align;
    int32_t point_channel_align;
    LocalTensor<T> h_im_local;
    LocalTensor<T> w_im_local;
    LocalTensor<int32_t> h_low_local;
    LocalTensor<int32_t> w_low_local;
    LocalTensor<int32_t> h_high_local;
    LocalTensor<int32_t> w_high_local;
    LocalTensor<T> h_low_t_local;
    LocalTensor<T> w_low_t_local;
    LocalTensor<T> lh_local;
    LocalTensor<T> lw_local;
    LocalTensor<T> neg_lh_local;
    LocalTensor<T> neg_lw_local;
    LocalTensor<T> hh_local;
    LocalTensor<T> hw_local;
    LocalTensor<int32_t> h_low_ptr_offset_local;
    LocalTensor<int32_t> h_high_ptr_offset_local;
    LocalTensor<int32_t> w_low_ptr_offset_local;
    LocalTensor<int32_t> w_high_ptr_offset_local;
    LocalTensor<T> w1_local;
    LocalTensor<T> w2_local;
    LocalTensor<T> w3_local;
    LocalTensor<T> w4_local;
    LocalTensor<T> grad_h_weight_local;
    LocalTensor<T> grad_w_weight_local;
    LocalTensor<T> top_grad_value_local;
    LocalTensor<T> v1_local;
    LocalTensor<T> v2_local;
    LocalTensor<T> v3_local;
    LocalTensor<T> v4_local;
    LocalTensor<T> v_w1_local;
    LocalTensor<T> v_w2_local;
    LocalTensor<T> mid_local;
    LocalTensor<T> w1_v1_local;
    LocalTensor<T> w2_v2_local;
    LocalTensor<T> w3_v3_local;
    LocalTensor<T> w4_v4_local;
    LocalTensor<T> val_local;
    LocalTensor<T> grad_weight_local;
    LocalTensor<T> attn_weight_local;
    LocalTensor<T> grad_output_local;
    LocalTensor<T> sample_location_local;
    LocalTensor<int32_t> level_start_index_local;
    LocalTensor<int32_t> spatial_shapes_local;
    LocalTensor<T> grad_weight_full_local;
    LocalTensor<T> grad_sample_loc_local;
};

template <typename T>
__aicore__ inline void MultiScaleDeformableAttentionGrad<T>::init(GM_ADDR input_tensors[INPUT_NUM + OUTPUT_NUM],
                                                                  MultiScaleDeformableAttentionGradTilingData *tiling_data)
{
    cur_block_idx = GetBlockIdx();
    spatial_size = tiling_data->spatial_size;
    num_heads = tiling_data->num_heads;
    channels = tiling_data->channels;
    num_levels = tiling_data->num_levels;
    num_query = tiling_data->num_query;
    num_point = tiling_data->num_point;

    int32_t doub = 2;
    int32_t batch_size = tiling_data->batch_size;
    cur_core_task_num = tiling_data->task_per_core;
    start_task_id = cur_core_task_num * cur_block_idx;
    if (cur_block_idx == tiling_data->core_used - 1)
    {
        cur_core_task_num = tiling_data->task_tail_core;
    }
    int32_t block_bytes = 32;
    t_per_block = block_bytes / sizeof(T);
    int32_per_block = block_bytes / sizeof(int32_t);
    value_gm.SetGlobalBuffer((__gm__ T *)(input_tensors[VALUE_INPUT_INDEX]), batch_size * spatial_size * num_heads * channels);
    spatial_shapes_gm.SetGlobalBuffer((__gm__ int32_t *)(input_tensors[SS_INPUT_INDEX]), num_levels);
    level_start_index_gm.SetGlobalBuffer((__gm__ int32_t *)(input_tensors[LSI_INPUT_INDEX]), num_levels * doub);
    sampling_loc_gm.SetGlobalBuffer((__gm__ T *)(input_tensors[SL_INPUT_INDEX]), batch_size * num_query * num_heads * num_levels * num_point * doub);
    attn_weight_gm.SetGlobalBuffer((__gm__ T *)(input_tensors[AW_INPUT_INDEX]), batch_size * num_query * num_heads * num_levels * num_point);
    grad_output_gm.SetGlobalBuffer((__gm__ T *)(input_tensors[GO_INPUT_INDEX]), batch_size * num_query * num_heads * channels);
    grad_value_gm.SetGlobalBuffer((__gm__ T *)(input_tensors[GV_OUTPUT_INDEX]), batch_size * spatial_size * num_heads * channels);
    grad_sampling_loc_gm.SetGlobalBuffer((__gm__ T *)(input_tensors[GSL_OUTPUT_INDEX]), batch_size * num_query * num_heads * num_levels * num_point * doub);
    grad_attn_weight_gm.SetGlobalBuffer((__gm__ T *)(input_tensors[GAW_OUTPUT_INDEX]), batch_size * num_query * num_heads * num_levels * num_point);
    num_point_align =  ceil(num_point, t_per_block);
    channel_align = ceil(channels, t_per_block);
    point_channel_align = ceil(num_point * channels, t_per_block);
    int32_t sampling_loc_size = num_heads * batch_size * num_levels * num_query * num_point;
}

template <typename T>
__aicore__ inline void MultiScaleDeformableAttentionGrad<T>::init_buffer()
{
    auto top_grad_ub_size = ceil(num_heads * channels, t_per_block);
    pipe.InitBuffer(in_queue_grad_output, 1, top_grad_ub_size * sizeof(T));
    pipe.InitBuffer(in_queue_lsi, 1, ceil(num_levels, int32_per_block) * sizeof(int32_t));
    pipe.InitBuffer(in_queue_ss, 1, ceil(num_levels * DOUB, int32_per_block) * sizeof(int32_t));
    pipe.InitBuffer(in_queue_sl, 1, ceil(num_heads * num_levels * num_point * DOUB * t_per_block,
                    t_per_block) * sizeof(T));
    pipe.InitBuffer(in_queue_aw, 1, ceil(num_heads * num_levels * num_point * t_per_block,
                    t_per_block) * sizeof(T));
    pipe.InitBuffer(buffer_h_im, num_point_align * sizeof(T));
    pipe.InitBuffer(buffer_w_im, num_point_align * sizeof(T));
    pipe.InitBuffer(buffer_h_low, num_point_align * sizeof(int32_t));
    pipe.InitBuffer(buffer_w_low, num_point_align * sizeof(int32_t));
    pipe.InitBuffer(buffer_h_high, num_point_align * sizeof(int32_t));
    pipe.InitBuffer(buffer_w_high, num_point_align * sizeof(int32_t));
    pipe.InitBuffer(buffer_h_low_t, num_point_align * sizeof(T));
    pipe.InitBuffer(buffer_w_low_t, num_point_align * sizeof(T));
    pipe.InitBuffer(buffer_lh, num_point_align * sizeof(T));
    pipe.InitBuffer(buffer_lw, num_point_align * sizeof(T));
    pipe.InitBuffer(buffer_neg_lh, num_point_align * sizeof(T));
    pipe.InitBuffer(buffer_neg_lw, num_point_align * sizeof(T));
    pipe.InitBuffer(buffer_hh, num_point_align * sizeof(T));
    pipe.InitBuffer(buffer_hw, num_point_align * sizeof(T));
    pipe.InitBuffer(buffer_h_low_ptr_offset, num_point_align * sizeof(int32_t));
    pipe.InitBuffer(buffer_h_high_ptr_offset, num_point_align * sizeof(int32_t));
    pipe.InitBuffer(buffer_w_low_ptr_offset, num_point_align * sizeof(int32_t));
    pipe.InitBuffer(buffer_w_high_ptr_offset, num_point_align * sizeof(int32_t));
    pipe.InitBuffer(buffer_w1, num_point_align * sizeof(T));
    pipe.InitBuffer(buffer_w2, num_point_align * sizeof(T));
    pipe.InitBuffer(buffer_w3, num_point_align * sizeof(T));
    pipe.InitBuffer(buffer_w4, num_point_align * sizeof(T));
    pipe.InitBuffer(buffer_grad_h_weight, point_channel_align * sizeof(T));
    pipe.InitBuffer(buffer_grad_w_weight, point_channel_align * sizeof(T));
    pipe.InitBuffer(buffer_top_grad_value, point_channel_align * sizeof(T));
    pipe.InitBuffer(buffer_v1, point_channel_align * sizeof(T));
    pipe.InitBuffer(buffer_v2, point_channel_align * sizeof(T));
    pipe.InitBuffer(buffer_v3, point_channel_align * sizeof(T));
    pipe.InitBuffer(buffer_v4, point_channel_align * sizeof(T));
    pipe.InitBuffer(buffer_v_w1, channel_align * sizeof(T));
    pipe.InitBuffer(buffer_v_w2, channel_align * sizeof(T));
    pipe.InitBuffer(buffer_mid, channel_align * sizeof(T));
    pipe.InitBuffer(buffer_w1_v1, channel_align * sizeof(T));
    pipe.InitBuffer(buffer_w2_v2, channel_align * sizeof(T));
    pipe.InitBuffer(buffer_w3_v3, channel_align * sizeof(T));
    pipe.InitBuffer(buffer_w4_v4, channel_align * sizeof(T));
    pipe.InitBuffer(buffer_val, point_channel_align * sizeof(T));
    pipe.InitBuffer(buffer_grad_weight, point_channel_align * sizeof(T));
    pipe.InitBuffer(buffer_grad_weight_full, (num_heads * num_levels * num_point * channel_align) * sizeof(T));
    pipe.InitBuffer(buffer_grad_sample_loc, (num_heads * num_levels * num_point * channel_align * DOUB) * sizeof(T));
}

template <typename T>
__aicore__ inline void MultiScaleDeformableAttentionGrad<T>::init_local_tensor()
{
    grad_output_local = in_queue_grad_output.AllocTensor<T>();
    level_start_index_local = in_queue_lsi.AllocTensor<int32_t>();
    spatial_shapes_local = in_queue_ss.AllocTensor<int32_t>();
    sample_location_local = in_queue_sl.AllocTensor<T>();
    attn_weight_local = in_queue_aw.AllocTensor<T>();
    h_im_local = buffer_h_im.Get<T>(num_point_align);
    w_im_local = buffer_w_im.Get<T>(num_point_align);
    h_low_local = buffer_h_low.Get<int32_t>(num_point_align);
    w_low_local = buffer_w_low.Get<int32_t>(num_point_align);
    h_high_local = buffer_h_high.Get<int32_t>(num_point_align);
    w_high_local = buffer_w_high.Get<int32_t>(num_point_align);
    h_low_t_local = buffer_h_low_t.Get<T>(num_point_align);
    w_low_t_local = buffer_w_low_t.Get<T>(num_point_align);
    lh_local = buffer_lh.Get<T>(num_point_align);
    lw_local = buffer_lw.Get<T>(num_point_align);
    neg_lh_local = buffer_neg_lh.Get<T>(num_point_align);
    neg_lw_local = buffer_neg_lw.Get<T>(num_point_align);
    hh_local = buffer_hh.Get<T>(num_point_align);
    hw_local = buffer_hw.Get<T>(num_point_align);
    h_low_ptr_offset_local = buffer_h_low_ptr_offset.Get<int32_t>(num_point_align);
    h_high_ptr_offset_local = buffer_h_high_ptr_offset.Get<int32_t>(num_point_align);
    w_low_ptr_offset_local = buffer_w_low_ptr_offset.Get<int32_t>(num_point_align);
    w_high_ptr_offset_local = buffer_w_high_ptr_offset.Get<int32_t>(num_point_align);
    w1_local = buffer_w1.Get<T>(num_point_align);
    w2_local = buffer_w2.Get<T>(num_point_align);
    w3_local = buffer_w3.Get<T>(num_point_align);
    w4_local = buffer_w4.Get<T>(num_point_align);
    grad_h_weight_local = buffer_grad_h_weight.Get<T>(point_channel_align);
    grad_w_weight_local = buffer_grad_w_weight.Get<T>(point_channel_align);
    top_grad_value_local = buffer_top_grad_value.Get<T>(point_channel_align);
    v1_local = buffer_v1.Get<T>(point_channel_align);
    v2_local = buffer_v2.Get<T>(point_channel_align);
    v3_local = buffer_v3.Get<T>(point_channel_align);
    v4_local = buffer_v4.Get<T>(point_channel_align);
    v_w1_local = buffer_v_w1.Get<T>(channel_align);
    v_w2_local = buffer_v_w2.Get<T>(channel_align);
    mid_local = buffer_mid.Get<T>(channel_align);
    w1_v1_local = buffer_w1_v1.Get<T>(channel_align);
    w2_v2_local = buffer_w2_v2.Get<T>(channel_align);
    w3_v3_local = buffer_w3_v3.Get<T>(channel_align);
    w4_v4_local = buffer_w4_v4.Get<T>(channel_align);
    val_local = buffer_val.Get<T>(point_channel_align);
    grad_weight_local = buffer_grad_weight.Get<T>(point_channel_align);
    grad_weight_full_local = buffer_grad_weight_full.Get<T>((num_heads * num_levels * num_point * channel_align) * sizeof(T));
    grad_sample_loc_local = buffer_grad_sample_loc.Get<T>((num_heads * num_levels * num_point * channel_align * DOUB) * sizeof(T));
}

template <typename T>
__aicore__ inline void MultiScaleDeformableAttentionGrad<T>::process()
{
    // SAVE FOR NEXT TILING MODE
    compute_mode_zero();
}

template <typename T>
__aicore__ inline void MultiScaleDeformableAttentionGrad<T>::process_grad_value_with_point(int32_t cur_nh, int32_t cur_nl,
                                                                                           int32_t base_ptr,
                                                                                           int32_t value_ptr_offset,
                                                                                           int32_t h, int32_t w)
{
    for (int32_t cur_np = 0; cur_np < num_point; cur_np++)
    {
        auto h_im = h_im_local.GetValue(cur_np);
        auto w_im = w_im_local.GetValue(cur_np);
        if ((float)-1.0 < h_im && h_im < (float)h && w_im > (float)-1.0 && w_im < (float)w)
        {
            auto attn_weight = attn_weight_local.GetValue((cur_nh * num_levels + cur_nl) * num_point + cur_np);
            set_flag(PIPE_S, PIPE_V, EVENT_ID1);
            wait_flag(PIPE_S, PIPE_V, EVENT_ID1);
            muls_template(top_grad_value_local[cur_np * channels], grad_output_local[cur_nh * channels],
                          attn_weight, channel_align);
            set_flag(PIPE_V, PIPE_S, EVENT_ID2);
            wait_flag(PIPE_V, PIPE_S, EVENT_ID2);
            auto h_low = h_low_local.GetValue(cur_np);
            auto h_high = h_high_local.GetValue(cur_np);
            auto w_low = w_low_local.GetValue(cur_np);
            auto w_high = w_high_local.GetValue(cur_np);
            if (h_low >= 0 && w_low >= 0)
            {
                cal_grad_value(v1_local, h_low_ptr_offset_local, w_low_ptr_offset_local, hw_local, hh_local,
                    w1_local, cur_np, base_ptr, value_ptr_offset, true, true);
            }
            if (h_low >= 0 && w_high < w)
            {
                cal_grad_value(v2_local, h_low_ptr_offset_local, w_high_ptr_offset_local, lw_local, hh_local,
                    w2_local, cur_np, base_ptr, value_ptr_offset, true, false);
            }
            if (h_high < h && w_low >= 0)
            {
                cal_grad_value(v3_local, h_high_ptr_offset_local, w_low_ptr_offset_local, hw_local, lh_local,
                    w3_local, cur_np, base_ptr, value_ptr_offset, false, true);
            }
            if (h_high < h && w_high < w)
            {
                cal_grad_value(v4_local, h_high_ptr_offset_local, w_high_ptr_offset_local, lw_local, lh_local,
                    w4_local, cur_np, base_ptr, value_ptr_offset, false, false);
            }
            process_grad_weight_with_point(cur_nh, cur_np);
        }

    }
}

template <typename T>
__aicore__ inline void MultiScaleDeformableAttentionGrad<T>::process_grad_weight_with_point(int32_t cur_nh,
                                                                                            int32_t cur_np)
{
    auto w1 = w1_local.GetValue(cur_np);
    auto w2 = w2_local.GetValue(cur_np);
    auto w3 = w3_local.GetValue(cur_np);
    auto w4 = w4_local.GetValue(cur_np);
    set_flag(PIPE_S, PIPE_V, EVENT_ID2);
    wait_flag(PIPE_S, PIPE_V, EVENT_ID2);
    muls_template(w1_v1_local, v1_local[cur_np * channels], w1, channel_align);
    muls_template(w2_v2_local, v2_local[cur_np * channels], w2, channel_align);
    muls_template(w3_v3_local, v3_local[cur_np * channels], w3, channel_align);
    muls_template(w4_v4_local, v4_local[cur_np * channels], w4, channel_align);
    pipe_barrier(PIPE_V);
    #ifndef __GET_CODE_CHANNEL__
    DataCopy(val_local[cur_np * channels], w1_v1_local, channel_align);
    #endif
    pipe_barrier(PIPE_V);
    Add(val_local[cur_np * channels], val_local[cur_np * channels], w2_v2_local, channel_align);
    pipe_barrier(PIPE_V);
    Add(val_local[cur_np * channels], val_local[cur_np * channels], w3_v3_local, channel_align);
    pipe_barrier(PIPE_V);
    Add(val_local[cur_np * channels], val_local[cur_np * channels], w4_v4_local, channel_align);
    pipe_barrier(PIPE_V);
    Mul(grad_weight_local[cur_np * channels], val_local[cur_np * channels], grad_output_local[cur_nh * channels],
        channel_align);
}

template <typename T>
__aicore__ inline void MultiScaleDeformableAttentionGrad<T>::pre_process_levels(int32_t cur_nh, int32_t cur_b,
                                                                                int32_t cur_nl, int32_t cur_q,
                                                                                int32_t sl_size, int32_t h_stride,
                                                                                int32_t w_stride, int32_t loc_h_offset,
                                                                                int32_t loc_w_offset, int32_t w, int32_t h,
                                                                                int32_t sample_location_offset)
{
    if (num_point % T_BLOCK != 0)
    {
        sample_location_offset = (cur_b * num_query + cur_q) * sl_size + (cur_nh * num_levels + cur_nl) * DOUB * num_point;
        #ifndef __GET_CODE_CHANNEL__
        DataCopy(sample_location_local, sampling_loc_gm[sample_location_offset], num_point_align);
        DataCopy(sample_location_local[num_point_align], sampling_loc_gm[sample_location_offset + num_point], num_point_align);
        #endif
        loc_w_offset = 0;
        loc_h_offset = num_point_align;
    }
    muls_template(h_im_local, sample_location_local[loc_h_offset], (float)h, num_point_align);
    adds_template(h_im_local, h_im_local, (float)(-0.5), num_point_align);
    muls_template(w_im_local, sample_location_local[loc_w_offset], (float)w, num_point_align);
    adds_template(w_im_local, w_im_local, (float)(-0.5), num_point_align);
    pipe_barrier(PIPE_V);
    Cast(h_low_local, h_im_local, RoundMode::CAST_FLOOR, num_point_align);
    Cast(w_low_local, w_im_local, RoundMode::CAST_FLOOR, num_point_align);
    pipe_barrier(PIPE_V);
    adds_template_int32(h_high_local, h_low_local, 1, num_point_align);
    adds_template_int32(w_high_local, w_low_local, 1, num_point_align);
    Cast(h_low_t_local, h_low_local, RoundMode::CAST_NONE, num_point_align);
    Cast(w_low_t_local, w_low_local, RoundMode::CAST_NONE, num_point_align);
    pipe_barrier(PIPE_V);
    Sub(lh_local, h_im_local, h_low_t_local, num_point_align);
    Sub(lw_local, w_im_local, w_low_t_local, num_point_align);
    pipe_barrier(PIPE_V);
    muls_template(neg_lh_local, lh_local, (float)-1.0, num_point_align);
    pipe_barrier(PIPE_V);
    adds_template(hh_local, neg_lh_local, (float)1.0, num_point_align);
    muls_template(neg_lw_local, lw_local, (float)-1.0, num_point_align);
    pipe_barrier(PIPE_V);
    adds_template(hw_local, neg_lw_local, (float)1.0, num_point_align);
    muls_template_int32(h_low_ptr_offset_local, h_low_local, h_stride, num_point_align);
    pipe_barrier(PIPE_V);
    adds_template_int32(h_high_ptr_offset_local, h_low_ptr_offset_local, h_stride, num_point_align);
    muls_template_int32(w_low_ptr_offset_local, w_low_local, w_stride, num_point_align);
    pipe_barrier(PIPE_V);
    adds_template_int32(w_high_ptr_offset_local, w_low_ptr_offset_local, w_stride, num_point_align);
    Mul(w1_local, hh_local, hw_local, num_point_align);
    Mul(w2_local, hh_local, lw_local, num_point_align);
    Mul(w3_local, lh_local, hw_local, num_point_align);
    Mul(w4_local, lh_local, lw_local, num_point_align);
    Duplicate<T>(grad_w_weight_local, 0.0, point_channel_align);
    Duplicate<T>(grad_h_weight_local, 0.0, point_channel_align);
    Duplicate<T>(v1_local, 0.0, point_channel_align);
    Duplicate<T>(v2_local, 0.0, point_channel_align);
    Duplicate<T>(v3_local, 0.0, point_channel_align);
    Duplicate<T>(v4_local, 0.0, point_channel_align);
}

template <typename T>
__aicore__ inline void MultiScaleDeformableAttentionGrad<T>::muls_template_int32(const LocalTensor<int32_t> &dstLocal,
                                                                                 const LocalTensor<int32_t> &srcLocal,
                                                                                 int32_t scalarValue, const int32_t calCount)
{
    int32_t unit = 256;
    int32_t max_repeat = 64;
    int32_t mask = unit / sizeof(int32_t);
    int32_t repeats = calCount / mask;
    int32_t loop = repeats / max_repeat;
    int32_t repeats_tail = repeats % max_repeat;
    int32_t tail = calCount % mask;
    int32_t tensor_offset = 0;
    for (int32_t loop_idx = 0; loop_idx < loop; loop_idx++)
    {
        Muls(dstLocal[loop_idx * max_repeat * mask], srcLocal[loop_idx * max_repeat * mask], scalarValue, mask, max_repeat,
             {DST_BLK_STRIDE, SRC_BLK_STRIDE, DST_REP_STRIDE, SRC_REP_STRIDE});
    }
    tensor_offset = loop * max_repeat * mask;
    if (repeats_tail >= 1)
    {
        Muls(dstLocal[tensor_offset], srcLocal[tensor_offset], scalarValue, mask, repeats_tail,
             {DST_BLK_STRIDE, SRC_BLK_STRIDE, DST_REP_STRIDE, SRC_REP_STRIDE});
    }
    tensor_offset += repeats_tail * mask;
    if (tail >= 1)
    {
        Muls(dstLocal[tensor_offset], srcLocal[tensor_offset], scalarValue, tail);
    }
}

template <typename T>
__aicore__ inline void MultiScaleDeformableAttentionGrad<T>::post_process_levels(int32_t w, int32_t h, int32_t cur_nl,
                                                                                 int32_t data_weight_ptr,
                                                                                 int32_t data_loc_w_ptr,
                                                                                 int32_t cur_nh)
{
    Mul(grad_w_weight_local, top_grad_value_local, grad_w_weight_local, point_channel_align);
    pipe_barrier(PIPE_V);
    muls_template(grad_w_weight_local, grad_w_weight_local, (float)w, point_channel_align);
    Mul(grad_h_weight_local, top_grad_value_local, grad_h_weight_local, point_channel_align);
    pipe_barrier(PIPE_V);
    muls_template(grad_h_weight_local, grad_h_weight_local, (float)h, point_channel_align);
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID3);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID3);
    Copy(grad_sample_loc_local[(cur_nh * num_levels + cur_nl) * num_point * 2 * channel_align], grad_w_weight_local, 64, num_point * channel_align / 64, { 1, 1, 8, 8 });
    Copy(grad_sample_loc_local[((cur_nh * num_levels + cur_nl) * 2 + 1) * num_point * channel_align], grad_h_weight_local, 64, num_point * channel_align / 64, { 1, 1, 8, 8 });
    Copy(grad_weight_full_local[(cur_nh * num_levels + cur_nl) * num_point * channel_align], grad_weight_local, 64, num_point * channel_align / 64, { 1, 1, 8, 8 });
}

template <typename T>
__aicore__ inline void MultiScaleDeformableAttentionGrad<T>::process_levels(int32_t cur_nh, int32_t base_ptr,
                                                                            int32_t cur_b, int32_t cur_q,
                                                                            int32_t sl_size, int32_t qid_stride,
                                                                            int32_t data_value_ptr_init_offset,
                                                                            int32_t w_stride,
                                                                            int32_t sample_location_offset,
                                                                            int32_t data_weight_ptr,
                                                                            int32_t data_loc_w_ptr)
{
    for (int32_t cur_nl = 0; cur_nl < num_levels; cur_nl++)
    {
        auto level_start_id = level_start_index_local.GetValue(cur_nl);
        auto value_ptr_offset = data_value_ptr_init_offset + level_start_id * qid_stride;
        auto h = spatial_shapes_local.GetValue(2 * cur_nl);
        auto w = spatial_shapes_local.GetValue(2 * cur_nl + 1);
        auto h_stride = w * w_stride;
        auto loc_w_offset = (cur_nh * num_levels + cur_nl) * DOUB * num_point;
        auto loc_h_offset = loc_w_offset + num_point;
        set_flag(PIPE_V, PIPE_MTE3, EVENT_ID3);
        wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID3);
        pre_process_levels(cur_nh, cur_b, cur_nl, cur_q, sl_size, h_stride,
                           w_stride, loc_h_offset, loc_w_offset, w, h, sample_location_offset);
        process_grad_value_with_point(cur_nh, cur_nl, base_ptr, value_ptr_offset, h, w);
        pipe_barrier(PIPE_V);
        post_process_levels(w, h, cur_nl, data_weight_ptr, data_loc_w_ptr, cur_nh);
    }
}

template <typename T>
__aicore__ inline void MultiScaleDeformableAttentionGrad<T>::compute_mode_zero()
{
    auto qid_stride = num_heads * channels;
    auto sl_size = num_heads * num_levels * num_point * DOUB;
    auto w_stride = num_heads * channels;
    #ifndef __GET_CODE_CHANNEL__
    DataCopy(level_start_index_local, level_start_index_gm, ceil(num_levels, int32_per_block));
    DataCopy(spatial_shapes_local, spatial_shapes_gm, ceil(num_levels * DOUB, int32_per_block));
    #endif
    for (int32_t b_nq_ind = start_task_id; b_nq_ind < start_task_id + cur_core_task_num; b_nq_ind++)
    {
        int32_t cur_q = b_nq_ind % num_query;
        int32_t cur_b = b_nq_ind / num_query;
        auto data_value_ptr_init_offset = cur_b * spatial_size * qid_stride;
        auto grad_output_offset = (cur_b * num_query + cur_q) * num_heads * channels;
        auto sample_location_offset = (cur_b * num_query + cur_q) * sl_size;
        #ifndef __GET_CODE_CHANNEL__
        DataCopy(grad_output_local, grad_output_gm[grad_output_offset], ceil(num_heads * channels, t_per_block));
        DataCopy(attn_weight_local, attn_weight_gm[sample_location_offset / DOUB], ceil(sl_size / DOUB, t_per_block));
        #endif
        if (num_point % T_BLOCK == 0)
        {
            #ifndef __GET_CODE_CHANNEL__
            DataCopy(sample_location_local, sampling_loc_gm[sample_location_offset], ceil(sl_size, t_per_block));
            #endif
        }
        auto data_weight_ptr = (cur_b * num_query * num_heads + cur_q * num_heads) * num_levels * num_point;
        auto data_loc_w_ptr = DOUB * data_weight_ptr;
        for (int32_t cur_nh = 0; cur_nh < num_heads; cur_nh++)
        {
            auto base_ptr = cur_nh * channels;
            process_levels(cur_nh, base_ptr, cur_b, cur_q, sl_size, qid_stride,
                           data_value_ptr_init_offset, w_stride,
                           sample_location_offset, data_weight_ptr, data_loc_w_ptr);
        }
        pipe_barrier(PIPE_V);
        int32_t time = 248;
        auto ran = num_heads * num_levels * num_point / time;
        auto ran1 = num_heads * num_levels * num_point * DOUB / time;
        auto remain = num_heads * num_levels * num_point % time;
        auto remain1 = num_heads * num_levels * num_point * DOUB % time;

        if (channels > 64)
        {
            auto mask = channels / t_per_block;
            ran = num_heads * num_levels * num_point * t_per_block / time;
            ran1 = num_heads * num_levels * num_point * DOUB * t_per_block / time;
            remain = num_heads * num_levels * num_point * t_per_block % time;
            remain1 = num_heads * num_levels * num_point * DOUB * t_per_block % time;
            for (auto i = 0; i < ran; i++)
            {
                WholeReduceSum<T>(attn_weight_local[i * time], grad_weight_full_local[i * time * mask], mask, time, 1, 1, (mask - 1) / t_per_block + 1);
            }
            for (auto i = 0; i < ran1; i++)
            {
                WholeReduceSum<T>(sample_location_local[i * time], grad_sample_loc_local[i * time * mask], mask, time, 1, 1, (mask - 1) / t_per_block + 1);
            }
            pipe_barrier(PIPE_V);
            WholeReduceSum<T>(attn_weight_local[ran * time], grad_weight_full_local[ran * time * mask], mask, remain, 1, 1, (mask - 1) / t_per_block + 1);
            WholeReduceSum<T>(sample_location_local[ran1 * time], grad_sample_loc_local[ran1 * time * mask], mask, remain1, 1, 1, (mask - 1) / t_per_block + 1);
            pipe_barrier(PIPE_V);

            ran = num_heads * num_levels * num_point / time;
            ran1 = num_heads * num_levels * num_point * DOUB / time;
            remain = num_heads * num_levels * num_point % time;
            remain1 = num_heads * num_levels * num_point * DOUB % time;
            for (auto i = 0; i < ran; i++)
            {
                WholeReduceSum<T>(attn_weight_local[i * time], attn_weight_local[i * time * t_per_block], t_per_block, time, 1, 1, 1);
            }
            for (auto i = 0; i < ran1; i++)
            {
                WholeReduceSum<T>(sample_location_local[i * time], sample_location_local[i * time * t_per_block], t_per_block, time, 1, 1, 1);
            }
            pipe_barrier(PIPE_V);
            WholeReduceSum<T>(attn_weight_local[ran * time], attn_weight_local[ran * time * t_per_block], t_per_block, remain, 1, 1, 1);
            WholeReduceSum<T>(sample_location_local[ran1 * time], sample_location_local[ran1 * time * t_per_block], t_per_block, remain1, 1, 1, 1);
            set_flag(PIPE_V, PIPE_MTE3, EVENT_ID3);
            wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID3);
        } else
        {
            for (auto i = 0; i < ran; i++)
            {
                WholeReduceSum<T>(attn_weight_local[i * time], grad_weight_full_local[i * time * channels], channels, time, 1, 1, (channels - 1) / t_per_block + 1);
            }
            pipe_barrier(PIPE_V);
            for (auto i = 0; i < ran1; i++)
            {
                WholeReduceSum<T>(sample_location_local[i * time], grad_sample_loc_local[i * time * channels], channels, time, 1, 1, (channels - 1) / t_per_block + 1);
            }
            pipe_barrier(PIPE_V);
            WholeReduceSum<T>(attn_weight_local[ran * time], grad_weight_full_local[ran * time * channels], channels, remain, 1, 1, (channels - 1) / t_per_block + 1);
            WholeReduceSum<T>(sample_location_local[ran1 * time], grad_sample_loc_local[ran1 * time * channels], channels, remain1, 1, 1, (channels - 1) / t_per_block + 1);
            set_flag(PIPE_V, PIPE_MTE3, EVENT_ID3);
            wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID3);
        }

        SetAtomicAdd<T>();
        #ifndef __GET_CODE_CHANNEL__
        DataCopyParams copy_params{1, (uint16_t)(num_heads * num_levels * num_point * sizeof(float)), 0, 0};
        DataCopyParams copy_params1{1, (uint16_t)(num_heads * num_levels * num_point * DOUB * sizeof(float)), 0, 0};
        DataCopyPad(grad_attn_weight_gm[data_weight_ptr], attn_weight_local, copy_params);
        DataCopyPad(grad_sampling_loc_gm[data_loc_w_ptr], sample_location_local, copy_params1);
        #endif
        SetAtomicNone();
        pipe_barrier(PIPE_ALL);
    }
    in_queue_grad_output.FreeTensor(grad_output_local);
    in_queue_lsi.FreeTensor(level_start_index_local);
    in_queue_ss.FreeTensor(spatial_shapes_local);
    in_queue_sl.FreeTensor(sample_location_local);
    in_queue_aw.FreeTensor(attn_weight_local);
}

template <typename T>
__aicore__ inline void MultiScaleDeformableAttentionGrad<T>::cal_grad_value(LocalTensor<T> &v_ub, LocalTensor<int32_t> &offset1_ub,
                                                                            LocalTensor<int32_t> &offset2_ub, LocalTensor<T> &h_w_w1_ub,
                                                                            LocalTensor<T> &h_w_w2_ub, LocalTensor<T> &w_weight_ub,
                                                                            int32_t cur_np, int32_t base_ptr, int32_t value_ptr_offset,
                                                                            bool neg_h, bool neg_w)
{
    auto ptr = offset1_ub.GetValue(cur_np) + offset2_ub.GetValue(cur_np) + base_ptr;
    auto h_w_w1 = h_w_w1_ub.GetValue(cur_np);
    auto h_w_w2 = h_w_w2_ub.GetValue(cur_np);
    auto w_weight = w_weight_ub.GetValue(cur_np);
    set_flag(PIPE_S, PIPE_MTE2, EVENT_ID0);
    wait_flag(PIPE_S, PIPE_MTE2, EVENT_ID0);
    
    #ifndef __GET_CODE_CHANNEL__
    DataCopy(v_ub[cur_np * channels], value_gm[value_ptr_offset + ptr], channel_align);
    #endif
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    muls_template(v_w1_local, v_ub[cur_np * channels], h_w_w1, channel_align);
    pipe_barrier(PIPE_V);
    if (neg_h)
    {
        Sub(grad_h_weight_local[cur_np * channels], grad_h_weight_local[cur_np * channels], v_w1_local, channel_align);
    } else
    {
        Add(grad_h_weight_local[cur_np * channels], grad_h_weight_local[cur_np * channels], v_w1_local, channel_align);
    }
    muls_template(v_w2_local, v_ub[cur_np * channels], h_w_w2, channel_align);
    pipe_barrier(PIPE_V);
    if (neg_w)
    {
        Sub(grad_w_weight_local[cur_np * channels], grad_w_weight_local[cur_np * channels], v_w2_local, channel_align);
    } else
    {
        Add(grad_w_weight_local[cur_np * channels], grad_w_weight_local[cur_np * channels], v_w2_local, channel_align);
    }
    muls_template(mid_local, top_grad_value_local[cur_np * channels], w_weight, channel_align);
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID3);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID3);
    SetAtomicAdd<T>();
    DataCopyParams copy_params3{1, (uint16_t)(channels * sizeof(float)), 0, 0};
    #ifndef __GET_CODE_CHANNEL__
    DataCopyPad(grad_value_gm[value_ptr_offset + ptr], mid_local, copy_params3);
    #endif
    SetAtomicNone();
}

template <typename T>
__aicore__ inline void MultiScaleDeformableAttentionGrad<T>::adds_template_int32(const LocalTensor<int32_t> &dstLocal,
                                                                                 const LocalTensor<int32_t> &srcLocal,
                                                                                 int32_t scalarValue, const int32_t calCount)
{
    int32_t unit = 256;
    int32_t max_repeat = 64;
    int32_t mask = unit / sizeof(int32_t);
    int32_t repeats = calCount / mask;
    int32_t loop = repeats / max_repeat;
    int32_t repeats_tail = repeats % max_repeat;
    int32_t tail = calCount % mask;
    int32_t tensor_offset = 0;
    for (int32_t loop_idx = 0; loop_idx < loop; loop_idx++)
    {
        Adds(dstLocal[loop_idx * max_repeat * mask], srcLocal[loop_idx * max_repeat * mask], scalarValue, mask, max_repeat, {DST_BLK_STRIDE, SRC_BLK_STRIDE, 
             DST_REP_STRIDE, SRC_REP_STRIDE});
    }
    tensor_offset = loop * max_repeat * mask;
    if (repeats_tail >= 1)
    {
        Adds(dstLocal[tensor_offset], srcLocal[tensor_offset], scalarValue, mask, repeats_tail,
             {DST_BLK_STRIDE, SRC_BLK_STRIDE, DST_REP_STRIDE, SRC_REP_STRIDE});
    }
    tensor_offset += repeats_tail * mask;
    pipe_barrier(PIPE_ALL);
    if (tail >= 1)
    {
        Adds(dstLocal[tensor_offset], srcLocal[tensor_offset], scalarValue, tail);
    }
}

template <typename T>
__aicore__ inline int32_t MultiScaleDeformableAttentionGrad<T>::ceil(int32_t a, int32_t b)
{
    if (b == 0)
    {
        return 0;
    }
    return ((a - 1) / b + 1) * b;
}

template <typename T>
__aicore__ inline void MultiScaleDeformableAttentionGrad<T>::muls_template(const LocalTensor<T> &dstLocal,
                                                                           const LocalTensor<T> &srcLocal,
                                                                           T scalarValue, const int32_t calCount)
{
    int32_t unit = 256;
    int32_t max_repeat = 64;
    int32_t mask = unit / sizeof(T);
    int32_t repeats = calCount / mask;
    int32_t loop = repeats / max_repeat;
    int32_t repeats_tail = repeats % max_repeat;
    int32_t tail = calCount % mask;
    int32_t tensor_offset = 0;
    for (int32_t loop_idx = 0; loop_idx < loop; loop_idx++)
    {
        Muls(dstLocal[loop_idx * max_repeat * mask], srcLocal[loop_idx * max_repeat * mask], scalarValue, mask, max_repeat,
             {DST_BLK_STRIDE, SRC_BLK_STRIDE, DST_REP_STRIDE, SRC_REP_STRIDE});
    }
    tensor_offset = loop * max_repeat * mask;
    if (repeats_tail >= 1)
    {
        Muls(dstLocal[tensor_offset], srcLocal[tensor_offset], scalarValue, mask, repeats_tail,
             {DST_BLK_STRIDE, SRC_BLK_STRIDE, DST_REP_STRIDE, SRC_REP_STRIDE});
    }
    tensor_offset += repeats_tail * mask;
    pipe_barrier(PIPE_ALL);
    if (tail >= 1)
    {
        Muls(dstLocal[tensor_offset], srcLocal[tensor_offset], scalarValue, tail);
    }
}

template <typename T>
__aicore__ inline void MultiScaleDeformableAttentionGrad<T>::adds_template(const LocalTensor<T> &dstLocal,
                                                                           const LocalTensor<T> &srcLocal,
                                                                           T scalarValue, const int32_t calCount)
{
    int32_t unit = 256;
    int32_t max_repeat = 64;
    int32_t mask = unit / sizeof(T);
    int32_t repeats = calCount / mask;
    int32_t loop = repeats / max_repeat;
    int32_t repeats_tail = repeats % max_repeat;
    int32_t tail = calCount % mask;
    int32_t tensor_offset = 0;
    for (int32_t loop_idx = 0; loop_idx < loop; loop_idx++)
    {
        Adds(dstLocal[loop_idx * max_repeat * mask], srcLocal[loop_idx * max_repeat * mask], scalarValue, mask, max_repeat,
             {DST_BLK_STRIDE, SRC_BLK_STRIDE, DST_REP_STRIDE, SRC_REP_STRIDE});
    }
    tensor_offset = loop * max_repeat * mask;
    if (repeats_tail >= 1)
    {
        Adds(dstLocal[tensor_offset], srcLocal[tensor_offset], scalarValue, mask, repeats_tail,
             {DST_BLK_STRIDE, SRC_BLK_STRIDE, DST_REP_STRIDE, SRC_REP_STRIDE});
    }
    tensor_offset += repeats_tail * mask;
    if (tail >= 1)
    {
        Adds(dstLocal[tensor_offset], srcLocal[tensor_offset], scalarValue, tail);
    }
}

// core func
extern "C" __global__ __aicore__ void multi_scale_deformable_attention_grad(GM_ADDR value_gm, GM_ADDR spatial_shapes_gm,
                                                                            GM_ADDR level_start_index_gm, GM_ADDR sampling_loc_gm,
                                                                            GM_ADDR attn_weight_gm, GM_ADDR grad_output_gm,
                                                                            GM_ADDR grad_value_gm, GM_ADDR grad_sampling_loc_gm,
                                                                            GM_ADDR grad_attn_weight_gm, GM_ADDR workspace,
                                                                            GM_ADDR tiling_data)
{
    GET_TILING_DATA(tiling_datas, tiling_data);
    GM_ADDR gm_tensor[INPUT_NUM + OUTPUT_NUM] = {value_gm, spatial_shapes_gm, level_start_index_gm,
        sampling_loc_gm, attn_weight_gm, grad_output_gm, grad_value_gm, grad_sampling_loc_gm, grad_attn_weight_gm};
    MultiScaleDeformableAttentionGrad<float> op32;
    op32.init(gm_tensor, &tiling_datas);
    op32.init_buffer();
    op32.init_local_tensor();
    op32.process();
}