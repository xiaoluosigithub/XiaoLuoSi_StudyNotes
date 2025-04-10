# 1、创建仓库

~~~ bash
git init
git add README.md
git config --global user.email "you@example.com"
git config --global user.name "Your Name"
git commit -m "first commit"
git branch -M main
git remote add origin https://github.com/test.git
git push -u origin main
~~~

每句话的含义如下：

1. `git init`：该命令用于在当前目录中初始化一个新的Git仓库。它会创建一个名为.git的隐藏文件夹，用于存储Git仓库的相关信息。
2. `git add README.md`：该命令将名为"README.md"的文件添加到Git的暂存区。暂存区是Git用来跟踪文件更改的一个中间区域。
3. `git config --global user.email `"you@example.com"：该命令用于设置Git的全局配置，其中user.email是你的邮箱地址。这个配置将与你的提交记录相关联。
4. `git config --global user.name "Your Name"`：该命令用于设置Git的全局配置，其中user.name是你的用户名。这个配置将与你的提交记录相关联。
5. `git commit -m "first commit"`：该命令用于将暂存区中的文件提交到Git仓库。-m选项后面的内容是提交的描述信息，用于解释本次提交的目的。
6. `git branch -M main`：该命令用于重命名当前分支。这里将当前分支重命名为"main"，这是GitHub默认的主分支名称。
7. `git remote add origin https://github.com/Wang-Phil/test.git`：该命令用于将本地仓库与远程GitHub仓库关联起来。origin是远程仓库的别名，https://github.com/Wang-Phil/test.git是远程仓库的URL。
8. `git push -u origin main`：该命令用于将本地仓库的内容推送到远程GitHub仓库。-u选项表示将本地的"main"分支与远程仓库的"main"分支关联起来。这样，在以后的推送中，你只需要运行git push命令即可。

# 2、仓库的日常使用

在仓库中写入其他文件后，要先将该文件推入暂存区，再push到仓库

~~~bash
git add .
git commit -m "XXXX"
git push
~~~



检出（Checkout）和提交（Commit）是Git中两个非常重要的概念。

检出是指将代码从Git仓库中取出并放到本地工作目录中，以便进行修改和开发。在Git中，我们可以使用 git checkout 命令来进行检出操作，例如：

```
git checkout master
```

上述命令表示将当前分支切换到master分支，并将master分支的代码检出到本地工作目录中。

提交是指将本地工作目录中的代码变更保存到Git仓库中，以便其他人可以查看和使用。在Git中，我们可以使用 git commit 命令来进行提交操作，例如：

```
git commit -m "Add new feature"
```

上述命令表示将本地工作目录中的代码变更提交到当前分支，并添加一条提交信息为“Add new feature”的记录。

需要注意的是，提交操作只是将代码变更保存到本地仓库中，并不会同步到远程仓库中。如果需要将本地仓库中的代码同步到远程仓库中，还需要使用 git push 命令进行推送操作。

**git pull已包含检出（checkout操作）**

pull 命令包含了两个操作：fetch 和 merge。

fetch 操作是将远程仓库的代码更新到本地仓库中，但是并不会将代码合并到当前分支中，也就是说并不会进行检出操作。

merge 操作是将本地仓库中的代码合并到当前分支中，也就是进行检出操作。

因此，pull 命令包含了 merge 操作，也就是包含了检出操作。执行 pull 命令会将远程仓库的代码更新到本地仓库中，并将更新后的代码合并到当前分支中，从而实现了检出操作。



# 3、Git中的“LF will be replaced by CRLF”警告详解

https://blog.csdn.net/taiyangdao/article/details/78629107

在写入`.ipynb`文件时，报了如下错误

~~~bash
warning: in the working copy of 'Untitled.ipynb', LF will be replaced by CRLF the next time Git touches it
~~~

原因如下：

在Windows中使用CRLF标识一行的结束，而在Linux/UNIX系统中只使用LF标识一行的结束。CRLF即Carriage-Return Line-Feed的缩写。

通常情况下，Git库不会自动修改文件内容，但是默认会将入库的文件的行尾符设置为LF，会将检出的文件的行尾符设置为CRLF。但是在执行如下操作时出现警告

~~~bash
C:\works\>git add mywebdav.conf
warning: LF will be replaced by CRLF in mywebdav.conf.
The file will have its original line endings in your working directory.
~~~

工作目录中的mywebdav.conf文件的行尾是LF，但是这里在即将入Git库之前，却将LF转换为CRLF。所以给出警告。该警告无伤大雅，因为在Git库中的该文件仍然以LF为行尾。



但是如何去除该警告呢？Git的CRLF相关特性与gitattributes文件中的设置相关。



在工作目录中，我们可以通过设置eol属性控制一个文件的行尾为CRLF或LF。我们也可以通过设置core.eol属性控制当前Git库中的所有文件的行尾为CRLF或LF。我们还可以设置core.autocrlf属性以覆盖core.eol属性的设置。如果要设置工作目录中的文件的行尾总是CRLF，而Git库中的文件的行尾总是LF，可以core.autocrlf=true。



我们一般希望远程仓库中的代码为LF，就用：` git config --global core.autocrlf input` 就ok了。

这是一个Git的配置命令，它的作用是告诉Git在检出代码时不要自动将行尾转换为CRLF（Windows风格的换行符），而是保留原来的LF（Unix风格的换行符）。

core.autocrlf input的意思是告诉Git在检出代码时不要自动将行尾转换为CRLF，而是保留原来的LF。这通常用于跨平台协作开发，以避免因行尾转换导致的代码变化和冲突。

https://www.cnblogs.com/yanglei-xyz/p/17744368.html



# 4、如何忽略已经被版本追踪的文件

问题核心在于**Git已经追踪过这些文件的历史记录**，因此即使后续在`.gitignore`中添加了忽略规则，也无法自动从仓库中移除已追踪的文件。

**Git的追踪机制**
`.gitignore`仅对**未被追踪的文件**生效。若文件已通过`git add`或`push`提交到仓库，后续修改`.gitignore`将无法自动取消追踪

**从Git缓存中移除文件（保留本地文件）**

```bash
# 删除整个文件夹的追踪记录（保留本地文件）
git rm -r --cached 文件夹路径/
# 提交变更
git commit -m "移除已忽略的文件夹"
# 推送至远程仓库
git push origin 分支名
```