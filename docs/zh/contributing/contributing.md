
[TOC]

# 开发DrivingSDK
新增算子和模型可分别阅读文档：
* [算子](../operator_adaption/from-scratch.md)
* [模型](../migration_tuning/model_optimization.md)

# PullRequest
## PR 创建流程
1. **Fork本项目的仓库**
2. **Clone到本地**
3. **创建开发分支**
4. **进行开发**
   - 编写代码
   - 添加测试
   - 更新文档
   - 确保代码通过本地测试
5. **提交代码**
6. **推送到 Fork 仓库**
7. **创建 Pull Request**
   - 访问gitcode仓库页面
   - 点击"Pull Request"或"合并请求"
   - 填写PR描述（见PR创建页面模板）

## PR最佳实践
1. **保持PR小规模**
   - 一次PR只解决一个问题
   - 便于评审和理解
   - 提高合并效率
   - 建议：单个PR不超过1000行（含测试）代码变更
2. **及时更新**
   - 定期同步上游主分支
   - 及时响应评审意见
   - 保持 PR 活跃
3. **清晰描述**
   - 详细描述变更原因和方式
   - 提供测试方法
   - 添加截图或示例（如适用）

## PR评审与合入规则
### 评审要求
1. **评审人员要求**
   - 评审人员必须熟悉相关代码领域
   - 评审人员不能是PR作者本人
2. **评审检查项**
   - ✅ 代码质量和风格
   - ✅ 功能正确性
   - ✅ 测试覆盖率（分支60%，行80%）
   - ✅ 文档完整性
   - ✅ 性能影响
   - ✅ 安全性
   - ✅ 向后兼容性
3. **CI 检查要求**
   - ✅ 所有 CI 检查必须通过
4. **无 Block 评论**
   - PR不能有任何未解决问题
### 合入规则
1. **Squash and Merge**
   - 将 PR 的所有提交合并为一个提交
   - 保持主分支历史清晰
   - 提交消息使用PR标题
2. **必须满足的条件**
   - ✅ 至少1位Maintainer或Committer的/lgtm，和1个/approve
3. **禁止的操作**
   - ❌ 禁止 Force Push 到主分支
   - ❌ 禁止合并自己的 PR（必须有他人评审）
### 合并权限
- **Maintainer**：可以合并任何PR
- **Committer**：可以合并任何PR
- **Contributor**：无合并权限，需要等待Maintainer或Committer合并

# Special Interest Group
## 例会
* 周期：每1个月举行一次例会，可通过[Ascend开源社区](https://meeting.ascend.osinfra.cn/)搜索、查看sig-DrivingSDK的会议链接。
* 申报议题：通过[sig-DrivingSDK Etherpad链接](https://etherpad.ascend.osinfra.cn/p/sig-DrivingSDK)进入共享文档，编辑申报议题。
* 参会人员：maintainer、committer、contributor等核心成员，其他对本SIG感兴趣的人员。
* 会议内容：讨论遗留问题和进展；当期申报的议题；需求评审、任务和优先级；需求规划和进展（roadmap）；新晋maintainer、committer准入评审。
* 会议归档：会议纪要位于[sig-DrivingSDK Etherpad链接](https://etherpad.ascend.osinfra.cn/p/sig-DrivingSDK)。

## 成员列表
[SIG成员列表](https://gitcode.com/Ascend/community/blob/master/MindSeriesSDK/sigs/DrivingSDK/sig-info.yaml)。