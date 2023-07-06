# workloads=infer-workload/infer-10/6214e9_2017-10-04_workload-1_infer-10.csv
AFE=0
AISS=1
policy=aryl
ARYL=1
log_file=0
num_nodes=16
num_gpus=4

workload_scale=small

exp_name=$policy-$workload_scale-AISS
workloads=./experiment/workload/$workload_scale.csv
output=./experiment/out/"$num_nodes"node-"$num_gpus"gpu/$workload_scale/$exp_name

nohup python3 simulator.py --policy $policy --workload $workloads --AFE $AFE --AISS $AISS --ARYL $ARYL \
--log_file $log_file --output $output --min-nodes $num_nodes --num-gpus $num_gpus > /dev/null 2>&1 &