# Temp

# Configure the environment
You can use the following command to configure the environment required for this work:
```python
pip install -r requirements.txt
```

# Dataset
You can change the image size to suit your needs by running the following command:
```python
cd Data_Convert
python Resize.py --folder path_to_your_folder/ --size 1024 1024
```

# Run Code
You can fine-tune SAM and perform pre-distillation and fine-tuning with the following instructions, but please note that you need to change the dataset and weight path and other parameter settings in the config.yaml file.
```python
cd Data_Convert
python finetune_sam.py
python distillation.py
python Sec_Distillation.py
```
