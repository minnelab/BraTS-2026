import SimpleITK as sitk
preds_folder = "/output"
out_folder = "/output"

from pathlib import Path
import os
import numpy as np
from scipy.ndimage import label

Path(out_folder).mkdir(parents=True, exist_ok=True)
for file in os.listdir(preds_folder):
    if file.endswith(".nii.gz"):
        pred = sitk.ReadImage(os.path.join(preds_folder, file))
        pred = sitk.GetArrayFromImage(pred)


        # Assume background is 0
        background_label = 0

        # Get spacing, needed for volume computation
        itk_img = sitk.ReadImage(os.path.join(preds_folder, file))
        spacing = itk_img.GetSpacing()  # (x, y, z)

        # Connected component analysis for each label except background
        output = pred.copy()
        unique_labels = np.unique(pred)
        for lbl in unique_labels:
            if lbl == background_label:
                continue
            mask = (pred == lbl)
            # label connected components within this mask
            labeled_array, num_features = label(mask)
            for cc in range(1, num_features + 1):
                cc_voxels = (labeled_array == cc)
                size_voxels = np.sum(cc_voxels)
                # Calculate volume in mm3
                volume_mm3 = size_voxels * np.prod(spacing)
                if volume_mm3 < 27:
                    print(f"Removing label {lbl} with volume {volume_mm3} mm3, and area {size_voxels} voxels in file {file}")
                    output[cc_voxels] = background_label

        # Optionally: overwrite prediction or write output image
        # If you want to save new segmentation (optional, adapt as needed)
        out_img = sitk.GetImageFromArray(output)
        out_img.CopyInformation(itk_img)
        sitk.WriteImage(out_img, os.path.join(out_folder, file))