## 分支与 worktree 工作流

所有代码改动必须在隔离的 git worktree 中完成。绝不在主 worktree 中修改已追踪
文件、切换分支或提交。

开始时：

1. 在主 worktree 拉取最新远端状态：
   `git fetch origin`
2. 创建 feature 分支与 worktree：
   `git worktree add .agent/worktrees/<name> -b agent/<issue-or-slug> origin/main`
   任务明确指定其他基线时使用该基线。
3. 进入 worktree 目录。此后所有文件编辑、测试运行和提交都在其中进行。

完成后：

4. 在 worktree 内提交、推送并创建 PR。

主 worktree 的操作仅限只读：`git fetch`、`git log`、`git status`、为获取上下文
而读取文件。