# workloads=infer-workload/infer-10/6214e9_2017-10-04_workload-1_infer-10.csv
AFE=0
AISS=1
policy=dp
log_file=0

num_nodes=16
num_gpus=4
workload_scale=small

workloads=./experiment/workload/$workload_scale.csv
exp_name="$policy"-$workload_scale-AISS
output=./experiment/out/"$num_nodes"node-"$num_gpus"gpu/$workload_scale/$exp_name

# nohup python3 simulator.py --workload $workloads --AFE $AFE --AISS $AISS --log_file $log_file \
# --output $output --min-nodes $num_nodes --num-gpus $num_gpus --policy $policy > /dev/null 2>&1 &

python3 simulator.py --workload $workloads --AFE $AFE --AISS $AISS --log_file $log_file \
--output $output --min-nodes $num_nodes --num-gpus $num_gpus --policy $policy

# python3 simulator.py --workload $workloads --AFE $AFE --AISS $AISS --log_file $log_file --policy $policy \
# --output $output --min-nodes $num_nodes --num-gpus $num_gpus > 2.log