#!/bin/bash
QUEUE="hpc"
WALLTIME="1:00"
CORES=4
MEMORY="8GB"

N_VALUES=(11 15 19 23 27 31)
RE_VALUES=(100 400 1000)

mkdir -p pack

i=0
for N in "${N_VALUES[@]}"; do
  for Re in "${RE_VALUES[@]}"; do
    jobname="Spectral-N${N}-Re${Re}"

    cat <<EOF > pack/job_${i}.lsf
#!/bin/bash
#BSUB -q ${QUEUE}
#BSUB -W ${WALLTIME}
#BSUB -n ${CORES}
#BSUB -R "rusage[mem=${MEMORY}]"
#BSUB -J ${jobname}

uv run python Experiments/Spectral-Solver/run_solver.py --N ${N} --Re ${Re}
EOF

    i=$((i+1))
  done
done
