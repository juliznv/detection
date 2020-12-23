import os
import shutil


set_name = 'train'
set_file = os.path.join('ImageSets', f'{set_name}.txt')
if not os.path.exists(set_name):
    os.mkdir(set_name)
img_files = open(set_file, 'r').readlines()
old_path = [os.path.join('JPEGImages', i[:-1]+'.jpg') for i in img_files]
new_path = [os.path.join(set_name, i[:-1]+'.jpg') for i in img_files]
for i in range(len(old_path)):
    shutil.copy(old_path[i], new_path[i])