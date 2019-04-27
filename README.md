# misp
Medical Image Segmentation Pipeline

**Note**: `DemoDataset` folder in this repo is just to let you know how I organized my training data, therefore there is no "**real**" file inside.  

Also, this folder structure is used as a *premise* in all my related scripts.

---

## Q & A:
1. How to create a subset of the original dataset for faster iterations?  
> Below is an example which I used to create my "smaller version" dataset.
```bash
python create_tiny_dataset.py -s CRCdataset/train -d tinyCRC/train -n 600

python create_tiny_dataset.py -s CRCdataset/valid -d tinyCRC/valid -n 100
```