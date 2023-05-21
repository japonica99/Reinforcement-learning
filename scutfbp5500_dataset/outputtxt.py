import os

a_folder_path = './attractive'
a_output_file = 'trainTarget.txt'

with open(a_output_file, 'w') as f:
    for file_name in os.listdir(a_folder_path):
        f.write(file_name + '\n')

u_folder_path = './unattractive'
u_output_file = 'trainSource.txt'

with open(u_output_file, 'w') as f:
    for file_name in os.listdir(u_folder_path):
        f.write(file_name + '\n')
