# 开发Driving SDK

新增算子和模型可分别阅读文档：

* [算子](./docs/zh/operator_adaption/from-scratch.md)
* [模型](./docs/zh/migration_tuning/model_optimization.md)

# PullRequest

## PR 创建流程

1. **Fork本项目的仓库**
2. **Clone到本地**
3. **创建开发分支**
4. **进行开发**
   * 编写代码
   * 添加测试
   * 更新文档
   * 使用 pre-commit 工具对代码和文档进行格式检查与修复（见下方[代码提交前检查](#代码提交前检查)）
   * 确保代码通过本地测试
5. **提交代码**
6. **推送到 Fork 仓库**
7. **创建 Pull Request**
   * 访问gitcode仓库页面
   * 点击“Pull Request”或“合并请求”
   * 填写PR描述（见PR创建页面模板）

## PR最佳实践

1. **保持PR小规模**
   * 一次PR只解决一个问题
   * 便于评审和理解
   * 提高合并效率
   * 建议：单个PR不超过1000行（含测试）代码变更
2. **及时更新**
   * 定期同步上游主分支
   * 及时响应评审意见
   * 保持 PR 活跃
3. **清晰描述**
   * 详细描述变更原因和方式
   * 提供测试方法
   * 添加截图或示例（如适用）

<a id="pre-commit-check"></a>

## 代码提交前检查

为了确保提交的代码和文档符合项目规范，推荐使用 pre-commit 工具在提交前自动进行格式检查和修复。pre-commit 是一款基于 Git 钩子（Git Hooks）的开源代码质量管控工具，在执行 git commit提交代码前，会自动完成代码校验、格式规范化、基础问题排查等工作，校验不通过则阻断提交流程，从源头保障代码规范统一、质量合规。

### 安装与配置

```shell
# 配置国内pip镜像源，提升安装下载速度，解决安装卡顿、超时等问题。
pip config set global.index-url https://mirrors.huaweicloud.com/repository/pypi/simple

# 安装 pre-commit
pip install pre-commit

# 在项目根目录安装 git hooks
pre-commit install
```

### 使用方式

```shell
# 对所有文件执行检查（首次使用或修改配置后推荐）
pre-commit run --all-files

#扫描Git暂存区内的改动文件，自动完成代码格式化与合规性检查，并完成格式化问题代码自动修复，其它未能自动修复的错误请参考提示人工修复。
git add .
pre-commit run

# 日常提交时，pre-commit 会在 git commit 时自动触发，仅检查变更文件
git add .
git commit -S -m "test"
```

pre-commit 会在每次 `git commit` 前自动运行配置的检查项，检查不通过将阻止提交，确保合入仓库的代码质量。

如果检查失败，pre-commit 会输出具体的错误信息。部分检查项（如代码格式化）会自动修复文件，此时只需重新 `git add` 修复后的文件并再次提交即可；对于无法自动修复的问题，请根据错误提示手动修正后再提交。若pre-commit run执行时出现安装组件失败， 优先检查本地配置文件 `.pre-commit-config.yaml` 是否与主库保持一致，不一致请同步更新。

详细的配置说明和可用 hooks 列表，请参考 [Ascend 社区 pre-commit 使用指南](https://gitcode.com/Ascend/community/blob/master/docs/contributor/pre-commit-guide.md)。

## PR评审与合入规则

### 评审要求

1. **评审人员要求**
   * 评审人员必须熟悉相关代码领域
   * 评审人员不能是PR作者本人
2. **评审检查项**
   * ✅ 代码质量和风格
   * ✅ 功能正确性
   * ✅ 测试覆盖率（分支60%，行80%）
   * ✅ 文档完整性
   * ✅ 性能影响
   * ✅ 安全性
   * ✅ 向后兼容性
3. **CI 检查要求**
   * ✅ 所有 CI 检查必须通过
4. **无 Block 评论**
   * PR不能有任何未解决问题

### 合入规则

1. **Squash and Merge**
   * 将 PR 的所有提交合并为一个提交
   * 保持主分支历史清晰
   * 提交消息使用PR标题
2. **必须满足的条件**
   * ✅ 至少1位Maintainer或Committer的/lgtm，和1个/approve
3. **禁止的操作**
   * ❌ 禁止 Force Push 到主分支
   * ❌ 禁止合并自己的 PR（必须有他人评审）

### 合并权限

* **Maintainer**：可以合并任何PR
* **Committer**：可以合并任何PR
* **Contributor**：无合并权限，需要等待Maintainer或Committer合并

# Special Interest Group

## 例会

* 周期：每1个月举行一次例会，可通过[Ascend开源社区](https://meeting.ascend.osinfra.cn/)搜索、查看sig-DrivingSDK的会议链接。
* 申报议题：通过[sig-DrivingSDK Etherpad链接](https://etherpad.ascend.osinfra.cn/p/sig-DrivingSDK)进入共享文档，编辑申报议题。
* 参会人员：maintainer、committer、contributor等核心成员，其他对本SIG感兴趣的人员。
* 会议内容：讨论遗留问题和进展；当期申报的议题；需求评审、任务和优先级；需求规划和进展（roadmap）；新晋maintainer、committer准入评审。
* 会议归档：会议纪要位于[sig-DrivingSDK Etherpad链接](https://etherpad.ascend.osinfra.cn/p/sig-DrivingSDK)。

## 成员列表

[SIG成员列表](https://gitcode.com/Ascend/community/blob/master/MindSeriesSDK/sigs/DrivingSDK/sig-info.yaml)。
