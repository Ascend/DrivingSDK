/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
 */
#include "kernel_operator.h"
#include "lib/matmul_intf.h"
using namespace AscendC;
constexpr int32_t BUFFER_NUM = 2;                                     // tensor num for each queue

class KernelBatchMatmulVector {
public:
    __aicore__ inline KernelBatchMatmulVector() {}

    TQue<QuePosition::VECIN, 1> inQueueProj, inQueuePts, inQueueFeature;
    TBuf<TPosition::VECCALC> projbuf, ptsbuf, pointsbuf, indicesoffsetbuf, indicespairbuf, tempgmbuf;
    TQue<QuePosition::VECOUT, 1> outQueueOUTPUT;
    GlobalTensor<DTYPE_PROJECTION_MAT> projectionMatGm;
    GlobalTensor<DTYPE_PTS_EXTEND> ptsExtendGm;
    GlobalTensor<DTYPE_POINT_2D> point2dGm;
    uint64_t core_used;
    uint64_t core_data;
    uint64_t copy_loop;
    uint64_t copy_tail;
    uint64_t last_copy_loop;
    uint64_t last_copy_tail;
    uint64_t available_ub_size;
    uint64_t totalresult;
    uint64_t ptstotal;
    uint64_t dim4;
    uint64_t dim5;
    LocalTensor<DTYPE_PROJECTION_MAT> proj_mat_ub;
    LocalTensor<DTYPE_PROJECTION_MAT> pts_ub;
    LocalTensor<DTYPE_PROJECTION_MAT> point2d_ub;
    DataCopyPadParams padParams{false, 0, 0, 0};
    int32_t total_kernel_size;
    int32_t data_each_block = 8;

    __aicore__ inline void Init(GM_ADDR projection_mat,
                                GM_ADDR pts_extend,
                                GM_ADDR point_2d,
                                GM_ADDR workspace,
                                BatchMatmulVectorTilingData *tiling_data, TPipe* pipe)
    {
        ASSERT(GetBlockNum() != 0 && "block dim can not be zeronumber!");
        this->core_used = tiling_data->core_used;
        this->core_data = tiling_data->core_data;
        this->copy_loop = tiling_data->copy_loop;
        this->copy_tail = tiling_data->copy_tail;
        this->last_copy_loop = tiling_data->last_copy_loop;
        this->last_copy_tail = tiling_data->last_copy_tail;
        this->available_ub_size = tiling_data->available_ub_size;
        this->totalresult = tiling_data->totalresult;
        this->ptstotal = tiling_data->ptstotal;
        this->dim4 = tiling_data->dim4;
        this->dim5 = tiling_data->dim5;
        projectionMatGm.SetGlobalBuffer((__gm__ DTYPE_PROJECTION_MAT*)projection_mat, this->totalresult);
        ptsExtendGm.SetGlobalBuffer((__gm__ DTYPE_PROJECTION_MAT*)pts_extend, this->ptstotal);
        point2dGm.SetGlobalBuffer((__gm__ DTYPE_PROJECTION_MAT*)point_2d, this->totalresult);
       
        pipe->InitBuffer(projbuf, this->available_ub_size * dim4 * sizeof(DTYPE_PROJECTION_MAT));
        pipe->InitBuffer(ptsbuf, this->available_ub_size * dim4 * sizeof(DTYPE_PROJECTION_MAT));
        pipe->InitBuffer(pointsbuf, this->available_ub_size * dim4 * sizeof(DTYPE_PROJECTION_MAT));
    }

    __aicore__ inline void Process()
    {
        uint32_t core_id = GetBlockIdx();
        uint64_t start_address = core_id * this->core_data;
        if (core_id >= this->core_used) {
            return;
        }
        if (core_id != (this->core_used -1)) {
            for (uint32_t i = 0; i < this->copy_loop; i++) {
                uint64_t address = start_address + i * this->available_ub_size;
                IndicesCompute(i, this->available_ub_size, address);
            }
            if (this->copy_tail != 0) {
                uint64_t address = start_address + this->copy_loop * this->available_ub_size;
                IndicesCompute(this->copy_loop, this->copy_tail, address);
            }
        } else {
            for (uint32_t i = 0; i < this->last_copy_loop; i++) {
                uint64_t address = start_address + i * this->available_ub_size;
                IndicesCompute(i, this->available_ub_size, address);
            }
            if (this->last_copy_tail != 0) {
                uint64_t address = start_address + this->last_copy_loop * this->available_ub_size;
                IndicesCompute(this->last_copy_loop, this->last_copy_tail, address);
            }
        }
    }

private:
    __aicore__ inline void IndicesCompute(int32_t progress, int32_t tensor_size, uint64_t address)
    {
        proj_mat_ub = projbuf.Get<DTYPE_PROJECTION_MAT>();
        pts_ub = ptsbuf.Get<DTYPE_PROJECTION_MAT>();
        point2d_ub = pointsbuf.Get<DTYPE_PROJECTION_MAT>();
        DataCopyPadParams propadParams{true, 0, 4, 0};
        DataCopyParams copyParams_proj_ub{1, (uint16_t)(tensor_size * dim4 * sizeof(DTYPE_PROJECTION_MAT)), 0, 0};
        DataCopyPad(pts_ub, ptsExtendGm[address * dim5], copyParams_proj_ub, padParams);
        DataCopyPad(proj_mat_ub, projectionMatGm[address * dim5], copyParams_proj_ub, padParams);
        PipeBarrier<PIPE_ALL>();
        Mul(point2d_ub, proj_mat_ub, pts_ub, tensor_size * dim5);
        PipeBarrier<PIPE_ALL>();
        DataCopyPad(point2dGm[address * dim4], point2d_ub, copyParams_proj_ub);
        PipeBarrier<PIPE_ALL>();
    }
};

extern "C" __global__ __aicore__ void batch_matmul_vector(GM_ADDR projection_mat, GM_ADDR pts_extend,
                                                        GM_ADDR point_2d,
                                                        GM_ADDR workspace, GM_ADDR tiling)
{
    GET_TILING_DATA(tiling_data, tiling);
    KernelBatchMatmulVector op;
    TPipe pipe;
    op.Init(projection_mat, pts_extend, point_2d, workspace, &tiling_data, &pipe);
    op.Process();
}