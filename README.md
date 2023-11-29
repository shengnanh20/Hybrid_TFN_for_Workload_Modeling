# Hybrid_TFN_for_Workload_Modeling
This repo is the official implementation for workload modeling with the hybrid tensor fusion network(TFN).
## Method
![image](https://github.com/shengnanh20/Hybrid_TFN_for_Workload_Modeling/blob/main/model.png)

## Requirements

* python 3.9
* pytorch 1.12.1

## Datasets

* [HP Omnicept Cognitive Load Database (HPO-CLD)](https://developers.hp.com/omnicept/hp-omnicept-cognitive-load-database-hpo-cld-%E2%80%93-developing-multimodal-inference-engine-detecting-real-time-mental-workload-vr)

## Training

* To train on cognitive dataset, you can run: 
```
python3 train.py 
```

## Testing

* To test the model which has been trained on the cognitive dataset, you can run the testing script as following:
```
python test.py 
```
Replace MODEL_PATH with your local path of the trained model.

