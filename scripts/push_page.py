import subprocess
import os
import shutil
import sys

massive_path = '/mnt/d/massive/'

#1.将当前目录切换到上一层目录
os.chdir(massive_path)

subprocess.run('git add .', shell=True)
commit_msg = sys.argv[1]
commit_command = 'git commit  -m ' + commit_msg
print(commit_command)

subprocess.run(commit_command, shell=True)

subprocess.run('git push', shell=True)

#3.更新代码git pull
print("==========git pull==========")
subprocess.run('git pull', shell=True)

#4.gitbook build
print("==========gitbook build==========")
subprocess.run('gitbook build', shell=True)

#5.将_book目录的内容拷贝book目录
print("==========copy _book==========")
subprocess.run('cp -r ./_book ../', shell=True)

#6.丢弃本地修改
print("==========git checkout . ==========")
subprocess.run('git checkout .', shell=True)

#7.git切换到gh-pages
print("==========git checkout gh-pages==========")
subprocess.run('git checkout gh-pages', shell=True)

#8.将当前的内容都删除，除了.git目录
print("==========delete directory==========")

# current_dir = os.getcwd()

# for item in os.listdir(current_dir):
#     item_path = os.path.join(current_dir, item)
#     if item == '.git':
#         continue
#     if os.path.isfile(item_path):
#         os.remove(item_path)
#     elif os.path.isdir(item_path):
#         shutil.rmtree(item_path)

#9.将book目录的内容都拷贝到当前目录
print("==========copy directory==========")
subprocess.run('cp -r ../_book/* ./', shell=True)

#10.git add .
print("==========git add、commit、push==========")
subprocess.run('git add .', shell=True)
subprocess.run('git commit -m 在页面展示', shell=True)
subprocess.run('git push', shell=True)


#11.git change to master
print("==========git change to master==========")
subprocess.run('git checkout master', shell=True)
