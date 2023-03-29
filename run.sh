export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export OMP_NUM_THREADS=1

n_process=8
PY=python3.7

pids=()
p=1
active_proc=0

RUNNAME=vmware-vmx
rm -rf ./$RUNNAME
ln -s `which $PY` ./$RUNNAME

wait_empty_processor(){
  echo "[SHELL] Wait empty processor $active_proc/$n_process"
  while [ $active_proc -ge $n_process ]
  do
    for j in $( seq 1 ${#pids[@]} )
    do
      #echo "check $j"
      if [ -z "${pids[$j]}" ]
      then
        echo "we have empty pids[${j}]"
      else
        if [ "${pids[$j]}" -ne -1 ]
        then
          if [ -z "`ps aux | awk '{print $2 }' | grep ${pids[$j]}`" ]
          then
            echo "[SHELL] $j:${pids[$j]} Finish $(date +"%T")"
            pids[$j]=-1
            let active_proc=$active_proc-1

          fi
        fi
      fi
    done
    sleep 5
  done
}

#algos="MMSAEA BO MSAEA MAMPSO DRNESO DREM WMMSAEA"
#seeds="1 2 3 4 5"
#vars="3 5 8 10"
#problems="Rastrigin Ackley Griewank Schwefel"

algos="MMSAEA BO MSAEA MAMPSO DRNESO DREM WMMSAEA"
seeds="1 2 3"
vars="5"
problems="Rastrigin Ackley Griewank Schwefel"

for s in $seeds
do
  for algo in $algos
  do
    for problem in $problems
    do
      for var in $vars
      do
        cmd="./$RUNNAME experiment.py --problems $problem --algorithms $algo --seeds $s --n-vars $var"
        echo $cmd
        $cmd > "${problem}_${algorithm}.log" &
        pids[$p]=$!
        echo "[SHELL]pids[${p}]=${pids[$p]} :$cmd Start $(date +"%T")"
        let p=$p+1
        let active_proc=$active_proc+1
        sleep 5
        wait_empty_processor
      done
    done
  done
done
