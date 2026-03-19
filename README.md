## rethinking_GA2Ms
The code for the paper: Rethingking GA2Ms: performance and out-of-distribution generalisation

## How to use

### Create PYTHON env
The requirements for the environment can be found in the requirements.txt file.
The code that runs the TALENT benchmark needs Tensorflow and pytorch
The rest needs Keras with JAX backend

### Download the TALENT data:
In the talent_benchmark folder:
```
curl -L -o data.zip 'https://drive.usercontent.google.com/download?id=1Rr2uZsZzabshoa0KX0uCne-qMWFpbyAo&export=download&confirm=t&uuid=efeac6c6-2f6e-4aa8-8adb-cdd812893147'
unzip data.zip
rm data.zip
```

### Set up the folder structure
The CoMA2G model is called ecmac in the code

There needs to be a _logs_ folder and a _results_model_ folder.
The defaults are _./talent_benchmark/.logs_ and _./talent_benchmark/results_model_ but all subsequent commands take these
paths as arguments so anything can be used.

So (for example):
```
cd talent_benchmark
mkdir logs
mkdir results_model
```

Different _logs_ and _results_model_ folders will be required for different runs of the different commands 
(e.g. on normal or ood data).

### Run the TALENT benchmark on the CoMA2G model
Use the _run_ecmac_on_talent_ Make command:
```
make run_ecmac_on_talent DATA_DIR=the_data_folder LOGS_DIR=the_logs_dir RESULTS_MODEL_DIR=the_results_model_dir
```
By default the TALENT benchmark will run the ecmac code 15 times (reps) and save in the logs all the results for each rep 
and their averages.

### Visualise the TALENT results
Use the _create_rankings_vis_ Make command:
```
make create_rankings_vis DATA_DIR=the_data_dir LOGS_DIR=the_logs_dir
```

### Visualise the splines of the CoGA2M model for a dataset
Use the _create_spline_vis_ Make command:
```
make create_spline_vis DATA_DIR=the_data_dir LOGS_DIR=the_logs_dir  RESULTS_MODEL_DIR=the_results_model_dir DATASET=the_dataset
```
For the paper the datasets used were: 
1. airfoil_self_noise
2. concrete_compressive_strength
3. fifa
4. stock

### Create the OOD Data
In the paper we used two algorithms to create the OOD data, one based on the kNNR and another on the vSVR algorithms.
The code automatically selects only datasets that are regression data, have no categorical features and have 30 or fewer
features.

To run the kNNR use the _make_knnr_ood_datasets_ Make command. 
```
make make_knnr_ood_datasets DATA_DIR=the_data_dir OOD_DATA_DIR=the_knnr_ood_data_dir
```
To run the vSVR use _make_vsvr_ood_datasets_
```
make make_vsvr_ood_datasets DATA_DIR=the_data_dir OOD_DATA_DIR=the_vsvr_ood_data_dir
```

### Run the TALENT benchmark on the OOD datasets for ecmac (CoMA2G) and LightGBM
Change the TALENT benchmark's _seed_num_ config (_talent_benchmark/TALENT/configs/classical_configs.json_) from 15 to 1.
There are already multiple reps in the OOD data (see the N column in Table 1).
```
seed_num: 1
```

Each OOD generation will create 4 separate dataset folders denoted by their r2_quartile (folders named r2_quartile_1,
r2_quartile_2, r2_quartile_3 and r2_quartile_4). 
One will be just a copy of the normal datasets (r2_quartile_1). The others are more and more broken datasets. In the
paper for comparison we use the r2_quartile_1 (R2_QUARTILE=1) and r2_quartile_4 (R2_QUARTILE=4)

Then run the TALENT benchmark on both quartile datasets using the _run_model_on_ood_ Make command once for the CoMA2G model
(MODEL_TYPE=ecmac) and once for the LightGBM (MODEL_TYPE=lightgbm) model. 

```
make run_model_on_ood MODEL_TYPE=the_model OOD_DATA_DIR=the_ood_data_dir LOGS_DIR=the_logs_dir  RESULTS_MODEL_DIR=the_results_model_dir R2_QUARTILE=the_quartile
```

Make sure you use the correct ood data folder and use separate logs and results_model folders for each model and each quartile.

### Calculate the comparisons
Use the above logs and the script in src/analysis/ood/adversarial_based/adv_based_datasets_comparison.py to create the
comparison DataFrames that we show in the paper

