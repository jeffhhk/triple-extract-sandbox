#!/bin/bash
adirScript=$( cd $( dirname "$0" ) && pwd )

# Starts a slurm job, and runs a slurm-specific tail follow program to observe the results
# Weird rules of slurm:
#   1) The launch script (first argument) must be a bash script.
#   2) The launch script will be copied before running, so it's $0 will be messed up.
#
# Since the launch script must be bash, but the real program is not bash, we know we must
# find at least one more script to keep running.  In order to mitigate $0 being useless for finding
# the next script, we make the convention of setting the cwd to adirScript.
#
# If this proves a problem, we can convert the "second layer" scripts like with_cwdrel to accept
# a parameter, either the first ordered parameter, or apparently slurm will pass environment variables.

adirLog="$HOME/slurmlog"
mkdir -p "$adirLog"

# Let's err on the side of canceling an unintended job instead of letting all useless jobs continue
njobs=$(squeue --user="$USER" --format=%A | tail -n+2 | wc -l)
if [[ ! "$njobs" -eq "0" ]]
then
    jobs=$(squeue --user="$USER" --format=%A | tail -n+2)
    echo Cancelling $njobs previous slurm jobs: $jobs
    scancel $jobs
fi

echo Requesting job on ${SBATCH_PARTITION}
# This version of sbatch seems to not respect SLURM_MEM_PER_CPU, so we pass it explicitly.
resp=$(cd "$adirScript"; sbatch --parsable --mem-per-cpu=${SLURM_MEM_PER_CPU} --output="$adirLog"/slurm-%A_0.out --error="$adirLog"/slurm-%A_0.out -- "$@")
jobid=$(printf "%s" ${resp} | cut -d: -f2)
# TODO?: real array id
aid=0
afileLog="$adirLog"/slurm-${jobid}_${aid}.out
touch "$afileLog"
echo logging at $afileLog

python "$adirScript/slurm_tail.py" "$jobid" "$afileLog"
