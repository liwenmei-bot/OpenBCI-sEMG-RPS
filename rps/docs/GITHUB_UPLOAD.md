# 将本目录上传 GitHub

仓库建议名称：`OpenBCI-RPS-sEMG`。初次整理可以先选 Private，确认内容后再决定可见性。

本目录已经有 README、依赖和 .gitignore。使用 Git 上传到新建空仓库时，GitHub 端先不要勾选自动生成 README、许可证或 .gitignore，避免出现两套初始历史。操作依据：[GitHub 官方导入本地代码说明](https://docs.github.com/en/migrations/importing-source-code/using-the-command-line-to-import-source-code/adding-locally-hosted-code-to-github)。

## 上传前

```powershell
python tools/check_repository.py
python -m unittest discover -s tests -v
git --version
```

未安装 Git 时，先安装 Git；也可以使用 GitHub Desktop 完成本地仓库的发布。GitHub 账户的命令行认证按 GitHub 提示完成，不把密码、令牌写进源码或发给别人。

## 新空仓库

在 GitHub 创建自己的空仓库，复制仓库 HTTPS 地址。在本项目根目录执行：

```powershell
git init -b main
git add .
git status --short
git diff --cached --stat
```

确认暂存内容只有程序、示例配置、说明和测试，再执行：

```powershell
git commit -m "Add OpenBCI four-channel sEMG RPS pipeline"
$repoUrl = Read-Host "粘贴你刚创建的空仓库 HTTPS 地址"
git remote add origin "$repoUrl"
git push -u origin main
```

若 Git 首次提交要求姓名和邮箱，请按本人身份配置本仓库的 `user.name`、`user.email` 后重新提交。上面的命令不替用户设置或虚构身份。

## 已有内容的仓库

已有 README、提交或分支时，先克隆该仓库，再将本目录中的文件复制进去，在新分支提交并按正常方式合并。不要用强制推送覆盖已有历史。

## 后续更新

```powershell
git status --short
git diff
git add 07_live_svm_rps_label_4ch.py README.md
git commit -m "Update live recognition"
git push
```

若需要发布实验成绩，在 README 中同时说明受试者数、试次数、数据划分和离线/实时评估方式。当前记录中的 91.67% 不是所有用户都能达到的通用准确率，也不是 robust 门控效果的结论。

