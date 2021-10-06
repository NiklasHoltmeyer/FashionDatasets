# Default DS Strucutre
```bash
DeepFashion2 Dataset
├───test
│   └───image
├───train
│   ├───annos
│   ├───annotations
│   └───image
└───validation
    ├───annos
    └───image

```

# Training

## Step 1) Create Image-Label pairs

```python
python tools/list_image_annotations_pairs.py --ds_path "{DeepFashion2_Dataset_Path}\train" --split 0.8 0.2 0.0
```

```bash
...
├───train
    ├───annos
    ├───annotations
    ├───image
    ├───test.txt
    ├───train.txt
    └───val.txt
...
```


## (Step 2 Resize Images)
```python
python tools/batch_transform_images.py --src_path "{DeepFashion2_Dataset_Path}\train" --dst_path
"{DeepFashion2_Dataset_Path}\train_256" --sub_folders "annotations" "images"
```

```bash
DeepFashion2 Dataset
...
├───train
│   ├───annos
│   ├───annotations
│   └───images
├───train_256
│   ├───annotations
│   └───images
...

```
