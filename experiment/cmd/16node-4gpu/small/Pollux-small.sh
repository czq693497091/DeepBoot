# workloads=infer-workload/infer-10/6214e9_2017-10-04_workload-1_infer-10.csv
INFER_SCHEDULER=0
AFE=0
AISS=0
policy=pollux
log_file=0
REPAIR_TRAIN=0
num_nodes=16
num_gpus=4
workload_scale=small

exp_name=Pollux-"$workload_scale"-worepair
workloads=./experiment/workload/$workload_scale.csv
output=./experiment/out/"$num_nodes"node-"$num_gpus"gpu/"$workload_scale"/$exp_name


# log_file=$output/simulator.log
python simulator.py --workload $workloads --AFE $AFE --AISS $AISS --log_file $log_file  \
--INFER_SCHEDULER $INFER_SCHEDULER --output $output --min-nodes $num_nodes --REPAIR_TRAIN $REPAIR_TRAIN \
--num-gpus $num_gpus 

# python simulator.py --workload $workloads --afe $afe --aiss $aiss \
# --output $output --min-nodes $num_nodes --num-gpus $num_gpus