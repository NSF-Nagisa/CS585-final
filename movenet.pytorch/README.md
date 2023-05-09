# Movenet
## How To Run

1.Download COCO dataset2017 from https://cocodataset.org/. (You need train2017.zip, val2017.zip and annotations.)

2.Make data to our data format.
```
python scripts/make_coco_data_17keypooints.py
```

3.You can add your own data to the same format.

4.After putting data at right place, you can start training
```
python train.py
```

5.After training finished, you need to change the test model path to test. Such as this in predict.py
```
run_task.modelLoad("output/xxx.pth")
```


6.Run predict.py to show predict result, or run evaluate.py to compute my acc on test dataset.
```
python predict.py
```
7.Convert to onnx.
```
python pth2onnx.py
```

