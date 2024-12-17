/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
 */
#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "kernel_utils.h"
#include "lib/matmul_intf.h"
#include <cmath>

using namespace AscendC;
using namespace std;
constexpr uint32_t BUFFER_NUM = 2;
const int32_t aligned_byte_num = 8;
const int32_t rois_info_num = 6;

class RoiAlignRotatedV2 {
public:
    __aicore__ inline RoiAlignRotatedV2()
    {}
    __aicore__ inline void Init(GM_ADDR input, GM_ADDR rois, GM_ADDR output, const RoiAlignRotatedV2TilingData *tiling_data)
    {
        ASSERT(GetBlockNum() != 0 && "block dim can not be zero!");

        tileNum = tiling_data->tileNum;
        blockDim = tiling_data->blockDim;
        ub_total_size = tiling_data->ub_total_size;
        batch_size = tiling_data->batch_size;
        channels = tiling_data->channels;
        channels_aligned = tiling_data->channels_aligned;
        input_h = tiling_data->input_h;
        input_w = tiling_data->input_w;
        rois_num_aligned = tiling_data->rois_num_aligned;
        tail_num = tiling_data->tail_num;
        spatial_scale = tiling_data->spatial_scale;
        sampling_ratio = tiling_data->sampling_ratio;
        pooled_height = tiling_data->pooled_height;
        pooled_width = tiling_data->pooled_width;
        aligned = tiling_data->aligned;
        clockwise = tiling_data->clockwise;
        rois_num_per_Lcore = tiling_data->rois_num_per_Lcore;
        rois_num_per_Score = tiling_data->rois_num_per_Score;
        Lcore_num = tiling_data->Lcore_num;
        Score_num = tiling_data->Score_num;
        input_buffer_size = tiling_data->input_buffer_size;
        blockIndex = GetBlockIdx();

        if (aligned == true) {
            offset_scalar = -0.5;
        } else {
            offset_scalar = static_cast<float>(0);
        }

        total_rois_num = rois_num_aligned - tail_num;
        output_shape = pooled_height * pooled_width;

        if (blockIndex < Lcore_num) {
            rois_num_per_core = rois_num_per_Lcore;
        } else {
            rois_num_per_core = rois_num_per_Score;
        }

        uint32_t ub_size_for_loop = (static_cast<uint32_t>(ub_total_size)) / 8;
        uint32_t roi_size = rois_info_num * sizeof(float);
        rois_num_per_loop_limit = ((ub_size_for_loop / roi_size) / aligned_byte_num) * aligned_byte_num;
        if (rois_num_per_loop_limit == 0) {
            loopCount = 0;
            rois_num_per_loop = 0;
        } else if (rois_num_per_core <= rois_num_per_loop_limit) {
            loopCount = 1;
            rois_num_per_loop = rois_num_per_core;
        } else {
            loopCount = (rois_num_per_core - rois_num_per_loop_limit) / rois_num_per_loop_limit + 1;
            rois_num_per_loop = rois_num_per_loop_limit;
        }

        uint32_t rois_buffer_size = rois_num_per_loop * sizeof(float);

        ASSERT(tileNum != 0 && "tile num can not be zero!");

        inputGM.SetGlobalBuffer((__gm__ DTYPE_INPUT *)input, static_cast<uint64_t>(batch_size) * channels * input_h * input_w);
        roisGM.SetGlobalBuffer((__gm__ DTYPE_ROIS *)rois, static_cast<uint64_t>(rois_num_aligned) * rois_info_num);
        outputGM.SetGlobalBuffer((__gm__ DTYPE_OUTPUT *)output, static_cast<uint64_t>(total_rois_num) * channels * pooled_height * pooled_width);

        pipe.InitBuffer(RoisQueueBatchIdx, BUFFER_NUM, rois_buffer_size);
        pipe.InitBuffer(RoisQueueCenterX, BUFFER_NUM, rois_buffer_size);
        pipe.InitBuffer(RoisQueueCenterY, BUFFER_NUM, rois_buffer_size);
        pipe.InitBuffer(RoisQueueWidth, BUFFER_NUM, rois_buffer_size);
        pipe.InitBuffer(RoisQueueHeight, BUFFER_NUM, rois_buffer_size);
        pipe.InitBuffer(RoisQueueTheta, BUFFER_NUM, rois_buffer_size);
        pipe.InitBuffer(PwBuffer, BUFFER_NUM, rois_buffer_size);
        pipe.InitBuffer(PhBuffer, BUFFER_NUM, rois_buffer_size);
        pipe.InitBuffer(BinSizeHBuffer, BUFFER_NUM, rois_buffer_size);
        pipe.InitBuffer(BinSizeWBuffer, BUFFER_NUM, rois_buffer_size);
        pipe.InitBuffer(BinGridSizeHBuffer, BUFFER_NUM, rois_buffer_size);
        pipe.InitBuffer(BinGridSizeWBuffer, BUFFER_NUM, rois_buffer_size);
        pipe.InitBuffer(GridHBuffer, BUFFER_NUM, rois_buffer_size);
        pipe.InitBuffer(GridWBuffer, BUFFER_NUM, rois_buffer_size);
        pipe.InitBuffer(SinThetaBuffer, BUFFER_NUM, rois_buffer_size);
        pipe.InitBuffer(CosThetaBuffer, BUFFER_NUM, rois_buffer_size);
        pipe.InitBuffer(RoiStartHBuffer, BUFFER_NUM, rois_buffer_size);
        pipe.InitBuffer(RoiStartWBuffer, BUFFER_NUM, rois_buffer_size);
        pipe.InitBuffer(CountTensorBuffer, BUFFER_NUM, rois_buffer_size);
        pipe.InitBuffer(GridMulBuffer, BUFFER_NUM, rois_buffer_size);
        pipe.InitBuffer(OutputValueBuffer, BUFFER_NUM, input_buffer_size);
        pipe.InitBuffer(CountChannelBuffer, BUFFER_NUM, input_buffer_size);
        pipe.InitBuffer(WeightBuffer, BUFFER_NUM, input_buffer_size);
        pipe.InitBuffer(ValueBuffer, BUFFER_NUM, input_buffer_size);
        pipe.InitBuffer(TmpValueBuffer, BUFFER_NUM, input_buffer_size);
        pipe.InitBuffer(AtomicAddBuffer, BUFFER_NUM, input_buffer_size);
        if (channels == channels_aligned) {
            pipe.InitBuffer(inQueueInput, BUFFER_NUM, input_buffer_size * 4);
        } else {
            pipe.InitBuffer(inQueueInput, BUFFER_NUM, input_buffer_size);
        }
    }

    __aicore__ inline void Process()
    {
        if (rois_num_per_core > 0) {
            OutputValue = OutputValueBuffer.AllocTensor<float>();
            CountChannel = CountChannelBuffer.AllocTensor<float>();
            WeightTensor = WeightBuffer.AllocTensor<float>();
            ValueTensor = ValueBuffer.AllocTensor<float>();
            TmpValueTensor = TmpValueBuffer.AllocTensor<float>();

            BinSizeH = BinSizeHBuffer.AllocTensor<float>();
            BinSizeW = BinSizeWBuffer.AllocTensor<float>();
            RoiBinGridH = BinGridSizeHBuffer.AllocTensor<float>();
            RoiBinGridW = BinGridSizeWBuffer.AllocTensor<float>();
            GridHTensor = GridHBuffer.AllocTensor<float>();
            GridWTensor = GridWBuffer.AllocTensor<float>();
            RoiSinTheta = SinThetaBuffer.AllocTensor<float>();
            RoiCosTheta = CosThetaBuffer.AllocTensor<float>();
            RoiStartH = RoiStartHBuffer.AllocTensor<float>();
            RoiStartW = RoiStartWBuffer.AllocTensor<float>();
            CountTensor = CountTensorBuffer.AllocTensor<float>();
            GridMulTensor = GridMulBuffer.AllocTensor<float>();
            
            Pw = PwBuffer.AllocTensor<float>();
            Ph = PhBuffer.AllocTensor<float>();
            AtomicAddTensor = AtomicAddBuffer.AllocTensor<float>();
            PipeBarrier<PIPE_V>();

            Duplicate(Pw, static_cast<float>(pooled_width), rois_num_per_loop);
            Duplicate(Ph, static_cast<float>(pooled_height), rois_num_per_loop);
            Duplicate(AtomicAddTensor, (float)1., channels_aligned);

            for (int32_t idx = channels; idx < channels_aligned; idx++) {
                AtomicAddTensor.SetValue(idx, (float)0.);
            }

            for (uint32_t i = 0; i < loopCount; i++) {
                RoisCopyIn(i, rois_num_per_loop);
                Compute(i, rois_num_per_loop);
            }

            OutputValueBuffer.FreeTensor<float>(OutputValue);
            CountChannelBuffer.FreeTensor<float>(CountChannel);
            WeightBuffer.FreeTensor<float>(WeightTensor);
            ValueBuffer.FreeTensor<float>(ValueTensor);
            TmpValueBuffer.FreeTensor<float>(TmpValueTensor);

            PwBuffer.FreeTensor<float>(Pw);
            PhBuffer.FreeTensor<float>(Ph);
            AtomicAddBuffer.FreeTensor<float>(AtomicAddTensor);

            BinSizeHBuffer.FreeTensor<float>(BinSizeH);
            BinSizeWBuffer.FreeTensor<float>(BinSizeW);
            BinGridSizeHBuffer.FreeTensor<float>(RoiBinGridH);
            BinGridSizeWBuffer.FreeTensor<float>(RoiBinGridW);
            GridHBuffer.FreeTensor<float>(GridHTensor);
            GridWBuffer.FreeTensor<float>(GridWTensor);
            SinThetaBuffer.FreeTensor<float>(RoiSinTheta);
            CosThetaBuffer.FreeTensor<float>(RoiCosTheta);
            RoiStartHBuffer.FreeTensor<float>(RoiStartH);
            RoiStartWBuffer.FreeTensor<float>(RoiStartW);
            CountTensorBuffer.FreeTensor<float>(CountTensor);
            GridMulBuffer.FreeTensor<float>(GridMulTensor);
            PipeBarrier<PIPE_ALL>();
        }
    }

private:
    __aicore__ inline void RoisCopyIn(uint32_t progress, int32_t rois_num)
    {
        LocalTensor<DTYPE_ROIS> RoisBatchIdx = RoisQueueBatchIdx.AllocTensor<DTYPE_ROIS>();
        LocalTensor<DTYPE_ROIS> RoisCenterX = RoisQueueCenterX.AllocTensor<DTYPE_ROIS>();
        LocalTensor<DTYPE_ROIS> RoisCenterY = RoisQueueCenterY.AllocTensor<DTYPE_ROIS>();
        LocalTensor<DTYPE_ROIS> RoisWidth = RoisQueueWidth.AllocTensor<DTYPE_ROIS>();
        LocalTensor<DTYPE_ROIS> RoisHeight = RoisQueueHeight.AllocTensor<DTYPE_ROIS>();
        LocalTensor<DTYPE_ROIS> RoisTheta = RoisQueueTheta.AllocTensor<DTYPE_ROIS>();
        PipeBarrier<PIPE_ALL>();

        if (blockIndex < Lcore_num) {
            uint64_t pre_idx = blockIndex * rois_num_per_core + progress * rois_num_per_loop;
            uint64_t cast_total_rois_num = static_cast<uint64_t>(total_rois_num);
            DataCopy(RoisBatchIdx, roisGM[pre_idx], rois_num);
            DataCopy(RoisCenterX, roisGM[pre_idx + cast_total_rois_num], rois_num);
            DataCopy(RoisCenterY, roisGM[pre_idx + cast_total_rois_num * 2], rois_num);
            DataCopy(RoisWidth, roisGM[pre_idx + cast_total_rois_num * 3], rois_num);
            DataCopy(RoisHeight, roisGM[pre_idx + cast_total_rois_num * 4], rois_num);
            DataCopy(RoisTheta, roisGM[pre_idx + cast_total_rois_num * 5], rois_num);
            PipeBarrier<PIPE_ALL>();
        } else {
            uint64_t pre_idx = static_cast<uint64_t>(Lcore_num) * rois_num_per_Lcore + (blockIndex - Lcore_num) * rois_num_per_core + progress * rois_num_per_loop;
            uint64_t cast_total_rois_num = static_cast<uint64_t>(total_rois_num);
            DataCopy(RoisBatchIdx, roisGM[pre_idx], rois_num);
            DataCopy(RoisCenterX, roisGM[pre_idx + cast_total_rois_num], rois_num);
            DataCopy(RoisCenterY, roisGM[pre_idx + cast_total_rois_num * 2], rois_num);
            DataCopy(RoisWidth, roisGM[pre_idx + cast_total_rois_num * 3], rois_num);
            DataCopy(RoisHeight, roisGM[pre_idx + cast_total_rois_num * 4], rois_num);
            DataCopy(RoisTheta, roisGM[pre_idx + cast_total_rois_num * 5], rois_num);
            PipeBarrier<PIPE_ALL>();
        }
        
        RoisQueueBatchIdx.EnQue<DTYPE_ROIS>(RoisBatchIdx);
        RoisQueueCenterX.EnQue<DTYPE_ROIS>(RoisCenterX);
        RoisQueueCenterY.EnQue<DTYPE_ROIS>(RoisCenterY);
        RoisQueueWidth.EnQue<DTYPE_ROIS>(RoisWidth);
        RoisQueueHeight.EnQue<DTYPE_ROIS>(RoisHeight);
        RoisQueueTheta.EnQue<DTYPE_ROIS>(RoisTheta);
        PipeBarrier<PIPE_ALL>();
    }

    __aicore__ inline void Compute(uint32_t progress, int32_t rois_num)
    {
        LocalTensor<DTYPE_ROIS> RoisBatchIdx = RoisQueueBatchIdx.DeQue<DTYPE_ROIS>();
        LocalTensor<DTYPE_ROIS> RoisCenterX = RoisQueueCenterX.DeQue<DTYPE_ROIS>();
        LocalTensor<DTYPE_ROIS> RoisCenterY = RoisQueueCenterY.DeQue<DTYPE_ROIS>();
        LocalTensor<DTYPE_ROIS> RoisWidth = RoisQueueWidth.DeQue<DTYPE_ROIS>();
        LocalTensor<DTYPE_ROIS> RoisHeight = RoisQueueHeight.DeQue<DTYPE_ROIS>();
        LocalTensor<DTYPE_ROIS> RoisTheta = RoisQueueTheta.DeQue<DTYPE_ROIS>();

        Muls(RoisCenterX, RoisCenterX, spatial_scale, rois_num);
        Muls(RoisCenterY, RoisCenterY, spatial_scale, rois_num);
        Muls(RoisWidth, RoisWidth, spatial_scale, rois_num);
        Muls(RoisHeight, RoisHeight, spatial_scale, rois_num);
        PipeBarrier<PIPE_V>();

        Adds(RoisCenterX, RoisCenterX, offset_scalar, rois_num);
        Adds(RoisCenterY, RoisCenterY, offset_scalar, rois_num);
        
        if (!aligned) {
            Maxs(RoisWidth, RoisWidth, (float)1.0, rois_num);
            Maxs(RoisHeight, RoisHeight, (float)1.0, rois_num);
        }
        
        if (clockwise) {
            Muls(RoisTheta, RoisTheta, (float)(-1.0), rois_num);
        }
        PipeBarrier<PIPE_V>();

        Muls(RoiStartH, RoisHeight, (float)(-0.5), rois_num);
        Muls(RoiStartW, RoisWidth, (float)(-0.5), rois_num);
        Div(BinSizeH, RoisHeight, Ph, rois_num);
        Div(BinSizeW, RoisWidth, Pw, rois_num);
        Sin(RoiSinTheta, RoisTheta);
        Cos(RoiCosTheta, RoisTheta);
        PipeBarrier<PIPE_V>();

        if (sampling_ratio > 0) {
            RoiBinGridH.SetSize(rois_num);
            Duplicate(RoiBinGridH, static_cast<float>(sampling_ratio), rois_num);
            Duplicate(RoiBinGridW, static_cast<float>(sampling_ratio), rois_num);
            PipeBarrier<PIPE_V>();
        } else {
            Ceil(RoiBinGridH, BinSizeH, rois_num);
            Ceil(RoiBinGridW, BinSizeW, rois_num);
            PipeBarrier<PIPE_V>();
        }
        
        Div(GridHTensor, BinSizeH, RoiBinGridH, rois_num);
        Div(GridWTensor, BinSizeW, RoiBinGridW, rois_num);
        Mul(GridMulTensor, RoiBinGridW, RoiBinGridH, rois_num);
        Maxs(CountTensor, GridMulTensor, (float)1., rois_num);
        PipeBarrier<PIPE_V>();

        int32_t output_index = ComputeOutputIndex(progress);

        for (uint32_t j = 0; j < rois_num; j++) {
            float batch_idx = RoisBatchIdx.GetValue(j);
            batch_idx = static_cast<int32_t>(batch_idx);

            if (output_index < (total_rois_num * pooled_height * pooled_width)) {
                ComputeItem(output_index, j, batch_idx, RoisCenterX.GetValue(j), RoisCenterY.GetValue(j));
            }
            output_index += output_shape;
        }
        
        RoisQueueBatchIdx.FreeTensor<float>(RoisBatchIdx);
        RoisQueueCenterX.FreeTensor<float>(RoisCenterX);
        RoisQueueCenterY.FreeTensor<float>(RoisCenterY);
        RoisQueueWidth.FreeTensor<float>(RoisWidth);
        RoisQueueHeight.FreeTensor<float>(RoisHeight);
        RoisQueueTheta.FreeTensor<float>(RoisTheta);
        PipeBarrier<PIPE_ALL>();
    }

    __aicore__ inline int32_t ComputeOutputIndex(uint32_t progress)
    {
        int32_t output_index;
        if (blockIndex < Lcore_num) {
            output_index = pooled_height * pooled_width * (blockIndex * rois_num_per_core + progress * rois_num_per_loop);
        } else {
            output_index = pooled_height * pooled_width * (Lcore_num * rois_num_per_Lcore + (blockIndex - Lcore_num) * rois_num_per_core + progress * rois_num_per_loop);
        }
        return output_index;
    }

    __aicore__ inline void ComputeItem(int32_t output_index, uint32_t roi_idx, int32_t batch_idx, float roi_center_w, float roi_center_h)
    {
        int32_t roi_bin_grid_h = RoiBinGridH.GetValue(roi_idx);
        int32_t roi_bin_grid_w = RoiBinGridW.GetValue(roi_idx);
        float bin_size_h = BinSizeH.GetValue(roi_idx);
        float bin_size_w = BinSizeW.GetValue(roi_idx);
        float grid_h = GridHTensor.GetValue(roi_idx);
        float grid_w = GridWTensor.GetValue(roi_idx);
        float roi_start_h = RoiStartH.GetValue(roi_idx);
        float roi_start_w = RoiStartW.GetValue(roi_idx);
        float sin_theta = RoiSinTheta.GetValue(roi_idx);
        float cos_theta = RoiCosTheta.GetValue(roi_idx);
        float count = CountTensor.GetValue(roi_idx);
        Duplicate(CountChannel, count, channels_aligned);
        
        for (int32_t index = output_index; index < (output_index + pooled_height * pooled_width); index++) {
            Duplicate(OutputValue, (float)0., channels_aligned);

            int32_t pw = index / pooled_width;
            int32_t ph = index / pooled_width / pooled_height;
            pw = index - pw * pooled_width;
            ph = (index / pooled_width) - ph * pooled_height;
        
            for (uint32_t iy = 0; iy < roi_bin_grid_h; iy++) {
                const float yy = roi_start_h + ph * bin_size_h + (iy + .5f) * grid_h;
                for (int ix = 0; ix < roi_bin_grid_w; ix++) {
                    const float xx = roi_start_w + pw * bin_size_w + (ix + .5f) * grid_w;

                    float y = yy * cos_theta - xx * sin_theta + roi_center_h;
                    float x = yy * sin_theta + xx * cos_theta + roi_center_w;

                    bilinear_interpolate(batch_idx, x, y, index);
                    ValueTensor = ValueBuffer.DeQue<float>();
                    PipeBarrier<PIPE_ALL>();
                    
                    Add(OutputValue, OutputValue, ValueTensor, channels);
                    PipeBarrier<PIPE_V>();
                }
            }

            Div(OutputValue, OutputValue, CountChannel, channels);
            PipeBarrier<PIPE_V>();

            if (channels != channels_aligned) {
                Mul(OutputValue, OutputValue, AtomicAddTensor, channels_aligned);
                PipeBarrier<PIPE_V>();

                OutputValueBuffer.EnQue<float>(OutputValue);
                PipeBarrier<PIPE_ALL>();
                
                SetAtomicAdd<float>();
                SingleRoiCopyOut(index);
                SetAtomicNone();
            } else {
                OutputValueBuffer.EnQue<float>(OutputValue);
                PipeBarrier<PIPE_ALL>();

                if (index == (output_index + pooled_height * pooled_width - 2)) {
                    Mul(OutputValue, OutputValue, AtomicAddTensor, channels_aligned);
                    PipeBarrier<PIPE_V>();
                }
                SingleRoiCopyOut(index);
            }
        }
    }

    __aicore__ inline void bilinear_interpolate(int32_t batch_idx, float x, float y, int32_t index)
    {
        if (y <  (float)-1.0 or y > input_h or x < (float)-1.0 or x > input_w) {
            Duplicate(ValueTensor, (float)0., channels_aligned);
            PipeBarrier<PIPE_ALL>();
        } else {
            if (y <= static_cast<float>(0)) {
                y = static_cast<float>(0);
            }
            if (x <= static_cast<float>(0)) {
                x = static_cast<float>(0);
            }
            int32_t x_floor = static_cast<int32_t>(x);
            int32_t y_floor = static_cast<int32_t>(y);
            int32_t x_ceil = x_floor + 1;
            int32_t y_ceil = y_floor + 1;
            
            if (x_floor >= (input_w - 1)) {
                x_ceil = input_w - 1;
                x_floor = x_ceil;
                x = static_cast<float>(x_ceil);
            }
            if (y_floor >= input_h - 1) {
                y_ceil = input_h - 1;
                y_floor = y_ceil;
                y = static_cast<float>(y_ceil);
            }

            float lx = x - static_cast<float>(x_floor);
            float ly = y - static_cast<float>(y_floor);
            float hx = static_cast<float>(1) - lx;
            float hy = static_cast<float>(1) - ly;

            if ((channels == channels_aligned) and (x_ceil > x_floor) and (y_ceil > y_floor)) {
                AlignedBilinearInterpolate(batch_idx, hx, hy, lx, ly, x_floor, y_floor, x_ceil, y_ceil);
            } else {
                NonAlignedBilinearInterpolate(batch_idx, hx, hy, lx, ly, x_floor, y_floor, x_ceil, y_ceil);
            }
        }
        ValueBuffer.EnQue<float>(ValueTensor);
        PipeBarrier<PIPE_ALL>();
    }

    __aicore__ void AlignedBilinearInterpolate(int32_t batch_idx, float hx, float hy, float lx, float ly, int32_t x_floor, int32_t y_floor, int32_t x_ceil, int32_t y_ceil)
    {
        LocalTensor<float> FeatureMap = inQueueInput.AllocTensor<float>();
        int32_t pre_idx = channels * (batch_idx * input_h * input_w);
        int32_t datacopy_idx = channels * (input_w * y_floor + x_floor) + pre_idx;
        PipeBarrier<PIPE_ALL>();
        AlignedSingleFeatureCopyIn(datacopy_idx, FeatureMap);

        FeatureMap = inQueueInput.DeQue<float>();

        float weight_p1 = hy * hx;
        float weight_p2 = hy * lx;
        float weight_p3 = ly * hx;
        float weight_p4 = ly * lx;

        Muls(TmpValueTensor, FeatureMap, weight_p1, channels);
        Muls(ValueTensor, FeatureMap[channels], weight_p2, channels);
        PipeBarrier<PIPE_V>();
        
        Add(ValueTensor, ValueTensor, TmpValueTensor, channels);
        PipeBarrier<PIPE_V>();

        Muls(TmpValueTensor, FeatureMap[channels * 2], weight_p3, channels);
        PipeBarrier<PIPE_V>();
        
        Add(ValueTensor, ValueTensor, TmpValueTensor, channels);
        PipeBarrier<PIPE_V>();

        Muls(TmpValueTensor, FeatureMap[channels * 3], weight_p4, channels);
        PipeBarrier<PIPE_V>();

        Add(ValueTensor, ValueTensor, TmpValueTensor, channels);
        PipeBarrier<PIPE_V>();
        inQueueInput.FreeTensor<float>(FeatureMap);
    }

    __aicore__ void NonAlignedBilinearInterpolate(int32_t batch_idx, float hx, float hy, float lx, float ly, int32_t x_floor, int32_t y_floor, int32_t x_ceil, int32_t y_ceil)
    {
        LocalTensor<float> FeatureMap = inQueueInput.AllocTensor<float>();
        int32_t pre_idx = channels * (batch_idx * input_h * input_w);
        float weight = hy * hx;
        int32_t datacopy_idx = channels * (input_w * y_floor + x_floor) + pre_idx;
        PipeBarrier<PIPE_ALL>();
        NonAlignedSingleFeatureCopyIn(datacopy_idx, channels_aligned, FeatureMap);
        FeatureMap = inQueueInput.DeQue<float>();
        Muls(TmpValueTensor, FeatureMap, weight, channels);
        PipeBarrier<PIPE_ALL>();
        
        weight = hy * lx;
        datacopy_idx = channels * (input_w * y_floor + x_ceil) + pre_idx;
        NonAlignedSingleFeatureCopyIn(datacopy_idx, channels_aligned, FeatureMap);
        FeatureMap = inQueueInput.DeQue<float>();
        Muls(ValueTensor, FeatureMap, weight, channels);
        PipeBarrier<PIPE_ALL>();
        
        weight = ly * hx;
        datacopy_idx = channels * (input_w * y_ceil + x_floor) + pre_idx;
        NonAlignedSingleFeatureCopyIn(datacopy_idx, channels_aligned, FeatureMap);
        FeatureMap = inQueueInput.DeQue<float>();
        Add(ValueTensor, ValueTensor, TmpValueTensor, channels);
        PipeBarrier<PIPE_ALL>();
        Muls(TmpValueTensor, FeatureMap, weight, channels);
        PipeBarrier<PIPE_ALL>();
        
        weight = ly * lx;
        datacopy_idx = channels * (input_w * y_ceil + x_ceil) + pre_idx;
        NonAlignedSingleFeatureCopyIn(datacopy_idx, channels_aligned, FeatureMap);
        FeatureMap = inQueueInput.DeQue<float>();
        Add(ValueTensor, ValueTensor, TmpValueTensor, channels);
        PipeBarrier<PIPE_ALL>();
        Muls(TmpValueTensor, FeatureMap, weight, channels);
        PipeBarrier<PIPE_ALL>();
        Add(ValueTensor, ValueTensor, TmpValueTensor, channels);
        PipeBarrier<PIPE_ALL>();
        inQueueInput.FreeTensor<float>(FeatureMap);
    }

    __aicore__ void AlignedSingleFeatureCopyIn(int32_t datacopy_idx, LocalTensor<float> FeatureMap)
    {
        DataCopyParams DataCopyParam = {
            (uint16_t)2,
            (uint16_t)(static_cast<uint16_t>(channels) * 2 / aligned_byte_num),
            (uint16_t)((static_cast<uint16_t>(channels) * (static_cast<uint16_t>(input_w) - 2)) / aligned_byte_num),
            (uint16_t)0
        };
        
        DataCopy(FeatureMap, inputGM[static_cast<uint64_t>(datacopy_idx)], DataCopyParam);
        PipeBarrier<PIPE_ALL>();
        inQueueInput.EnQue<float>(FeatureMap);
        PipeBarrier<PIPE_ALL>();
    }

    __aicore__ void NonAlignedSingleFeatureCopyIn(int32_t datacopy_idx, int32_t datacopy_len, LocalTensor<float> FeatureMap)
    {
        DataCopy(FeatureMap, inputGM[static_cast<uint64_t>(datacopy_idx)], datacopy_len);
        PipeBarrier<PIPE_ALL>();
        inQueueInput.EnQue<float>(FeatureMap);
        PipeBarrier<PIPE_ALL>();
    }

    __aicore__ inline void SingleRoiCopyOut(int32_t index)
    {
        OutputValue = OutputValueBuffer.DeQue<float>();
        PipeBarrier<PIPE_ALL>();
        DataCopy(outputGM[static_cast<uint64_t>(index) * channels], OutputValue, channels_aligned);
        PipeBarrier<PIPE_ALL>();
    }

private:
    TPipe pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM> RoisQueueBatchIdx, RoisQueueCenterX, RoisQueueCenterY, RoisQueueHeight, RoisQueueWidth, RoisQueueTheta;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueueInput;
    TQue<QuePosition::VECOUT, BUFFER_NUM> CountChannelBuffer, ValueBuffer, OutputValueBuffer;
    TQue<QuePosition::VECCALC, BUFFER_NUM> PwBuffer, PhBuffer, WeightBuffer, TmpValueBuffer, AtomicAddBuffer;
    TQue<QuePosition::VECCALC, BUFFER_NUM> BinSizeHBuffer, BinSizeWBuffer, BinGridSizeHBuffer, BinGridSizeWBuffer, GridHBuffer, GridWBuffer, GridSizeHBuffer, SinThetaBuffer, CosThetaBuffer, RoiStartHBuffer, RoiStartWBuffer, CountTensorBuffer, GridMulBuffer;

    GlobalTensor<DTYPE_INPUT> inputGM;
    GlobalTensor<DTYPE_ROIS> roisGM;
    GlobalTensor<DTYPE_OUTPUT> outputGM;

    LocalTensor<float> Ph, Pw, BinSizeH, BinSizeW, RoiBinGridH, RoiBinGridW, GridHTensor, GridWTensor, RoiSinTheta, RoiCosTheta, RoiStartH, RoiStartW, CountTensor, GridMulTensor;
    LocalTensor<float> RoiOutput, OutputValue, CountChannel;
    LocalTensor<float> WeightTensor, ValueTensor, TmpValueTensor, AtomicAddTensor;

    bool aligned, clockwise;
    uint32_t blockDim, tileNum, batch_size, channels, channels_aligned, input_h, input_w, rois_num_aligned, tail_num, total_rois_num;
    uint32_t rois_num_per_core, rois_num_per_loop, rois_num_per_loop_limit, loopCount;
    uint32_t rois_num_per_Lcore, rois_num_per_Score, Lcore_num, Score_num, input_buffer_size;
    int32_t sampling_ratio, pooled_height, pooled_width, output_shape;
    float spatial_scale, offset_scalar;
    uint64_t ub_total_size, blockIndex;
};

extern "C" __global__ __aicore__ void roi_align_rotated_v2(GM_ADDR input, GM_ADDR rois, GM_ADDR output, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);
    RoiAlignRotatedV2 op;
    op.Init(input, rois, output, &tiling_data);
    if (TILING_KEY_IS(1)) {
        op.Process();
    }
}