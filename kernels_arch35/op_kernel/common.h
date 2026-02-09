#ifndef COMMON_H_
#define COMMON_H_

#include "kernel_operator.h"

constexpr int32_t TILING_ALIGN32B_FLAG = 1;
constexpr int32_t TILING_FP32_BIT = 1;
constexpr int32_t TILING_FP16_BIT = 2;
constexpr int32_t TILING_BF16_BIT = 3;

class TaskIterator {
public:
    __aicore__ inline TaskIterator(
        int32_t blkIdx, int32_t blkDim, int32_t avgTaskNum, int32_t tailTaskNum, int32_t totalTaskNum)
        : blkIdx_(blkIdx), blkDim_(blkDim), totalTaskNum_(totalTaskNum)
    {
        nextIdx_ = blkIdx * avgTaskNum + (blkIdx < tailTaskNum ? blkIdx : tailTaskNum);
        endIdx_ = nextIdx_ + avgTaskNum + (blkIdx < tailTaskNum ? 1 : 0);
    }

    __aicore__ inline bool HasNext() const
    {
        return nextIdx_ < endIdx_;
    }

    __aicore__ inline int32_t Next()
    {
        return nextIdx_++;
    }

    __aicore__ inline int32_t GetNext() const
    {
        return nextIdx_;
    }

    __aicore__ inline int32_t GetTaskNum() const
    {
        return totalTaskNum_;
    }

private:
    int32_t blkIdx_, blkDim_;
    int32_t nextIdx_, endIdx_;
    int32_t totalTaskNum_;
};
#endif // COMMON_H_