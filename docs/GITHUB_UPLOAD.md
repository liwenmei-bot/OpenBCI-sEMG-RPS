# 上传GitHub

这是整理后的源码目录，不包含受试者原始记录、拟合模型或实验输出。先保留原实验目录，把新目录作为仓库。若通过`prepare_openbci_semg_github.py`生成，目标必须是尚不存在的文件夹，生成器不会覆盖已有项目，也不会自动安装依赖或访问GitHub。

## 本地检查

```powershell
conda activate bci
Set-Location "D:\肌电\news\OpenBCI-sEMG-RPS"
python run.py check --help-checks --verify-snapshot
python -m unittest discover -s tests -v
python -m unittest discover -s rps/tests -v
```

如果缺依赖，在适合当前Python版本的环境中安装`requirements.txt`。已有模型继续使用与其训练兼容的环境。PsychoPy只在实际运行采集提示时需要。

## 建立仓库并推送

先在自己的GitHub账号下新建空仓库，例如`OpenBCI-sEMG-RPS`，按需要选择可见性；无需让GitHub另行生成README、gitignore或许可证，因为本地已经有README和gitignore。许可由代码维护者自行选择。

下面仅是命令模板，`<你的仓库HTTPS地址>`必须换成刚创建仓库的真实地址：

```powershell
git init -b main
git add .
git status --short
git diff --cached --stat
git commit -m "Organize OpenBCI four-channel sEMG gesture recognition"
git remote add origin "<你的仓库HTTPS地址>"
git push -u origin main
```

若Git要求提交身份，填写自己的`user.name`与`user.email`。如果当前目录已经是仓库，先检查`git status`及`git remote -v`，不必重复初始化或替换现有origin。GitHub认证使用自己已配置的方式，不在代码或命令文本里写Token。

提交前核对`git status`：应包含源码、JSON示例、README、docs、tests与CI；本地`data/`、`models/`、`outputs/`只提交空目录占位文件。`.gitignore`不会自动移除过去已跟踪的数据；现有仓库需要另外检查跟踪清单。

## 以后更新

```powershell
git status --short
git add README.md docs binary rps tools tests configs run.py requirements.txt requirements-tested.txt requirements-collection.txt
git diff --cached --stat
git commit -m "Update analysis and live recognition workflow"
git push
```

初始SHA清单用于核对这次交付的源码。之后正常修改源码会令初始哈希不同，以Git差异、测试结果与实验记录管理更新，不要为了让旧哈希通过而隐藏改动。
