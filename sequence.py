import subprocess

program_list = ['preprocess.py', 'downsample.py', 'train.py', 'predict.py','interpolate.py']

for program in program_list:
    subprocess.call(['python', 'program'])
    print("Finished:" + program)