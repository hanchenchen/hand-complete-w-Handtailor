import subprocess


def Worker(exe_file, proc_id):
    subprocess.Popen(exe_file)
