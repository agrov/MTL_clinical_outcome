
# Multi-task Learning using AdapterFusion


### Pair-Wise HPO experiments using Hard parameter sharing method

Please use [ray-cluster.yaml](https://github.com/agrov/MTL_clinical_outcome/blob/main/experiments/k8/ray-cluster.yaml) to start the cluster and [ray-job.yaml](https://github.com/agrov/MTL_clinical_outcome/blob/main/experiments/k8/ray-job.yaml) to submit a job after making the necessary changes like the image name, namespace, etc. The sample ray-job.yaml file starts HPO for Diagnosis and Length of stay task using [hpo_dia_los.py](https://github.com/DATEXIS/MTL_clinical_outcome/blob/main/experiments/hpo_dia_los.py).  
Additionally, please make the necessary changes like log directory path in the corresponding [config file](https://github.com/agrov/MTL_clinical_outcome/tree/main/experiments/configs) before submitting the job.

### HPO for Single Task Adapters

[hpo_mp_sta.py](https://github.com/agrov/MTL_clinical_outcome/blob/main/experiments/Adapters/STA/hpo_mp_sta.py) includes the HPO for Mortality Prediction using Adapter Modules. The same python file can be used for other tasks as well by changing the [config file](https://github.com/agrov/MTL_clinical_outcome/tree/main/experiments/Adapters/config)  and log directory for ray.

### HPO for AdapterFusion

[fusion_mp.py](https://github.com/agrov/MTL_clinical_outcome/blob/main/experiments/Adapters/Fusion/fusion_mp.py) includes the HPO for Mortality Prediction using AdapterFusion. 
