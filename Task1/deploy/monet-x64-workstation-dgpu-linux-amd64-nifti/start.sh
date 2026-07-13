#!/bin/bash

cp /input/*-t1c.nii.gz /opt/data/Task101_BraTS-MET-2026/raw_splitted/imagesTs/$(basename /input/*-t1c.nii.gz | sed 's/-t1c.nii.gz/_0000.nii.gz/')
cp /input/*-t1n.nii.gz /opt/data/Task101_BraTS-MET-2026/raw_splitted/imagesTs/$(basename /input/*-t1n.nii.gz | sed 's/-t1n.nii.gz/_0001.nii.gz/')
cp /input/*-t2f.nii.gz /opt/data/Task101_BraTS-MET-2026/raw_splitted/imagesTs/$(basename /input/*-t2f.nii.gz | sed 's/-t2f.nii.gz/_0002.nii.gz/')
cp /input/*-t2w.nii.gz /opt/data/Task101_BraTS-MET-2026/raw_splitted/imagesTs/$(basename /input/*-t2w.nii.gz | sed 's/-t2w.nii.gz/_0003.nii.gz/')

/opt/venv/nndet/bin/nndet_predict 101 RetinaUNetV001_D3V001_3d --fold -1
/opt/venv/nndet/bin/nndet_boxes2nii 101 RetinaUNetV001_D3V001_3d --fold -1 --test --threshold 0.0
/opt/venv/nndet/bin/nndet_seg2nii 101 RetinaUNetV001_D3V001_3d --fold -1 --test
/opt/venv/nndet/bin/python /opt/code/nnDetection/MONet_concatenate_modalities.py

find /opt/models/Task101_BraTS-MET-2026/RetinaUNetV001_D3V001_3d/consolidated/test_predictions_nii -type f -name '*_image.nii.gz' -exec cp {} /var/holoscan/input/ \;
/var/holoscan/tools
for file in /var/holoscan/output/*_image_seg.nii.gz; do
    if [[ -f "$file" ]]; then
        newname="${file/_image_seg.nii.gz/.nii.gz}"
        mv "$file" "$newname"
        cp "$newname" /output/
    fi
done