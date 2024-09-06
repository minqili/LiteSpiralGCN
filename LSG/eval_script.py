import subprocess

input_dir = r'E:\HandMesh_after2\mobrecon\out\FreiHAND\mrc_ds_GCN_drop0.5_1'
output_dir = r'C:\Users\29785\Desktop\ok\mrc_ds_GCN_drop0.5_1'
pred_file_name = 'mrc_ds_GCN_drop0.5_18.json'

# 构建要执行的命令
command = ['python', r'E:\HandMesh_after2\mobrecon\freihand-master\eval.py', input_dir, output_dir, '--pred_file_name', pred_file_name]

# 执行命令运行
subprocess.run(command)
