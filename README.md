# Temp

# Configure the environment
You can use the following command to configure the environment required for this work:
```cpp
pip install -r requirements.txt
```

# Dataset
You can get the DiffuseMix data augmentation data, including the original annotation data, through the following link
[DiffuseMix_enhance](https://drive.google.com/drive/folders/1jg1K7Jnt1aOSqfuJOC4ttRvGSqp7UP_j?usp=drive_link)

You can get the enhanced data of Albu through the following link, including the original annotated data
[Albu](https://drive.google.com/drive/folders/17q--4qp-LSA-8snQwQRD1dzdAr3oLPEH?usp=drive_link)

You can change the image size to suit your needs by running the following command:
```cpp
cd Data_Convert
python Resize.py --folder path_to_your_folder/ --size 1024 1024
```
You can download these two datasets and combine them in any proportion according to your needs.

# Weight
From the following link you can get the fine-tuned SAM weights and the distilled SegNet weights:
[SAM](https://drive.google.com/file/d/1zHW4oEcvleprqaUdaD2MC4kLloAnuHF-/view?usp=drive_link)
[SegNet](https://drive.google.com/file/d/1OnxCtapiQyxbFfU4yEKJFOT-_4IiwEsA/view?usp=drive_link)

# Run Code
You can fine-tune SAM and perform pre-distillation and fine-tuning with the following instructions, but please note that you need to change the dataset and weight path and other parameter settings in the config.yaml file.
```cpp
python finetune_sam.py
python distillation.py
python Sec_Distillation.py
```
