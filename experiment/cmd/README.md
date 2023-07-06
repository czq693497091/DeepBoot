# CMD to start the simulator

## Name rules of the bash:

$method-$workload_scale-$params

## Methods

- DeepBoot
- Pollux (OSDI'21): https://www.usenix.org/conference/osdi21/presentation/qiao
- AFS (NSDI'21): https://www.usenix.org/conference/nsdi21/presentation/hwang
- Tiresias (NSDI'19): https://www.usenix.org/conference/nsdi19/presentation/gu
- Optimus (EuroSys'18): https://dl.acm.org/doi/10.1145/3190508.3190517
- Ayrl: https://arxiv.org/abs/2202.07896 (Lyra, EuroSys'23, https://dl.acm.org/doi/10.1145/3552326.3587445)

## Parameters

- policy: Method. dp = DeepBoot.
- AFE: Whether use AFE to reduce the cost during allocation update. 1 means selected.
- AISS: Equal to ATS-I, providing the inference task schedule and lifecycle for inference task.
- protect_times: Times to the base protect time. Base protect time is 120 and the bonus is 15. If $protect_times=2. protect time is 240 and bonus is 30, etc.
- workload_scale: small, middle, large. pure_train workload only contains training DLTs.

## Run the bash

```bash
bash experiment/cmd/16node-4gpu/small/xxx.sh
```







