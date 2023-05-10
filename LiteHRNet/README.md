# Lite HRNet-dark Human Body 2D Pose Estimation

This project is code based on the MMPose .

## Environment Install

After downloading the code, enter mmpose folder
Run the following code:
pip install -r requirements.txt
pip install -v -e .
This code might change based on your cuda:
pip install mmcv -f https://download.openmmlab.com/mmcv/dist/cu118/torch2.0.0/index.html
pip install mmengine

## Data preparation

Enter datasets/coco/images folder, create folder if the folder doesn't exist.

### Train

To train the model
Download coco dataset with following code:
wget http://images.cocodataset.org/zips/train2017.zip
unzip -qq train2017.zip
rm train2017.zip

Then run the following code to train:
python tools/train.py configs/litehrnet-dark_coco-256x192.py --work-dir workDIR --auto-scale-lr --show-dir showDir

### Test

To test the model
Download coco dataset with following code:
wget http://images.cocodataset.org/zips/val2017.zip
unzip -qq val2017.zip
rm val2017.zip

Then run the following code to test:
python tools/test.py configs/litehrnet-dark_coco-256x192.py workDIR/LiteHRNet_dark.pth --work-dir workDIR --show-dir showDir --show --interval 1000
