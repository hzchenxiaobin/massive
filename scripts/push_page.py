import subprocess


#1.进入到上一级目录massive
subprocess.run('cd ..', shell=True)

#2.将分支切换到master
subprocess.run('git checkout master', shell=True)

#3.更新代码git pull
subprocess.run('git pull', shell=True)

#4.gitbook build
subprocess.run('gitbook build', shell=True)

#5.将_book目录的内容拷贝book目录
subprocess.run('cp -r ./book ../', shell=True)

#6.git切换到gh-pages

#6.将当前的内容都删除，除了.git目录

#7.将book目录的内容都拷贝到当前目录

#8.git add .

#9.git commit

#10.git push


