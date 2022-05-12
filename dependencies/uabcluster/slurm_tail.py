import time
import sys
import os
import fcntl
import subprocess

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

file = open(fn)
fcntl.fcntl(file, fcntl.F_SETFL, os.O_RDONLY|os.O_NONBLOCK)

doContinue=True
print("about to monitor {}".format(fn))
sys.stdout.flush()
while doContinue:
    iOffset = file.tell()
    line = file.readline()
    if not line:
        time.sleep(1)
        file.seek(iOffset)
        doContinue=isRunning(jobid)
    else:
        line=line.rstrip("\n")
        print(line)
        sys.stdout.flush()
