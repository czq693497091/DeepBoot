# workloads=infer-workload/infer-10/6214e9_2017-10-04_workload-1_infer-10.csv
AFE=0
AISS=1 # AISS有无对结果影响不大
policy=pollux
log_file=1

num_nodes=8
num_gpus=4
# workload_num=30

exp_name=Pollux-pure_train
workloads=./experiment/workload/pure_train.csv
output=./experiment/out/16node-4gpu/baseline/$exp_name

python simulator.py --workload $workloads --AFE $AFE --AISS $AISS --log_file $log_file \
--output $output --min-nodes $num_nodes --num-gpus $num_gpus