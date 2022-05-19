import time
import sys
import os
import fcntl
import subprocess
import signal

def isRunning(jobid):
    # capture_output=True only created in 3.7 (documentation lies):
    #   https://stackoverflow.com/questions/53209127/subprocess-unexpected-keyword-argument-capture-output
    res = subprocess.run(["squeue", "--jobs", str(jobid)], stdout=subprocess.PIPE, stderr=subprocess.PIPE, encoding='UTF-8')
    if len(res.stderr) > 0:
        raise Exception("could not detect slurm job", res)
    if sum(1 for l in res.stdout.split("\n") if len(l)>0) >= 2:
        return True
    return False

jobid=int(sys.argv[1])
fn=sys.argv[2]

def handler(signum, frame):
    print("Ctrl-c was pressed. Kill slurm job? y/n ", flush=True)
    res = input("")
    if res.lower().__contains__('y'):
        res = subprocess.run(["scancel", str(jobid)], stdout=subprocess.PIPE, stderr=subprocess.PIPE, encoding='UTF-8')
        print("scancel res={}".format(res))
        exit(0)
    else:
        print("allowing job {} to continue unattended".format(jobid), flush=True)
        exit(0)
signal.signal(signal.SIGINT, handler)

file = open(fn)
fcntl.fcntl(file, fcntl.F_SETFL, os.O_RDONLY|os.O_NONBLOCK)

cMorePolls = 3
print("about to monitor {}".format(fn))
sys.stdout.flush()
while cMorePolls>0:
    iOffset = file.tell()
    line = file.readline()
    if not line:
        time.sleep(1)
        file.seek(iOffset)
        if isRunning(jobid):
            cMorePolls = 3
        cMorePolls -= 1
    else:
        line=line.rstrip("\n")
        print(line)
        sys.stdout.flush()
