import subprocess
import os

#1.将当前目录切换到上一层目录
os.chdir('..')

#2.将分支切换到master
print("1.change branch to master.....")
subprocess.run('git checkout master', shell=True)

#3.更新代码git pull
print("2.git pull.....")
subprocess.run('git pull', shell=True)

#4.gitbook build
print("3.gitbook build.....")
subprocess.run('gitbook build', shell=True)

#5.将_book目录的内容拷贝book目录
print("4.copy _book.....")
subprocess.run('cp -r ./_book ../', shell=True)

#6.git切换到gh-pages
print("5.git checkout gh-pages.....")
subprocess.run('git checkout gh-pages', shell=True)

#6.将当前的内容都删除，除了.git目录

#7.将book目录的内容都拷贝到当前目录

#8.git add .

#9.git commit

#10.git push


