#!/usr/bin/env python

from __future__ import annotations

from pathlib import Path

try:
    import monai
    from monai.transforms import Compose, ConcatItemsd, EnsureChannelFirst, LoadImage, SaveImage, SpatialResample
except ImportError:
    SpatialResample = ConcatItemsd = LoadImage = SaveImage = EnsureChannelFirst = Compose = None
    monai = None

try:
    import numpy as np
except ImportError:
    np = None

try:
    import torch
except ImportError:
    torch = None
from typing import Any, Dict


def define_affine_from_meta(meta: Dict[str, Any]) -> np.ndarray:
    """
    Define an affine matrix from the metadata of a tensor.

    Parameters
    ----------
    meta : Dict[str, Any]
        Metadata dictionary containing 'pixdim', 'origin', and 'direction'.

    Returns
    -------
    np.ndarray
        A 4x4 affine matrix constructed from the metadata.
    """
    pixdim = meta["pixdim"]
    origin = meta["origin"]
    direction = meta["direction"].reshape(3, 3)

    # Extract 3D spacing
    spacing = pixdim[1:4]  # drop the first element (usually 1 for time dim)

    # Scale the direction vectors by spacing to get rotation+scale part
    affine = direction * spacing[np.newaxis, :]

    # Append origin to get 3x4 affine matrix
    affine = np.column_stack((affine, origin))

    # Make it a full 4x4 affine
    return torch.Tensor(np.vstack((affine, [0, 0, 0, 1])))


class NIFTINameFormatter:
    def __init__(self, suffix):
        self.suffix = suffix

    def __call__(self, metadict: dict, saver) -> dict:
        """Returns a kwargs dict for :py:meth:`FolderLayout.filename`,
        according to the input metadata and SaveImage transform."""
        subject = (
            metadict.get(monai.utils.ImageMetaKey.FILENAME_OR_OBJ, getattr(saver, "_data_index", 0))
            if metadict
            else getattr(saver, "_data_index", 0)
        )
        patch_index = metadict.get(monai.utils.ImageMetaKey.PATCH_INDEX, None) if metadict else None
        subject = subject[: -len(self.suffix)] + ".nii.gz"
        # subject = subject[:-len(self.filename_key)]+".nii.gz"
        return {"subject": f"{subject}", "idx": patch_index}


def get_arg_parser():
    import argparse

    parser = argparse.ArgumentParser(description="Concatenate modalities from Multi-Modal dataset.")
    parser.add_argument("--input_folder", type=str, required=True, help="Path to the input folder containing the dataset.")
    parser.add_argument(
        "--output_folder", type=str, required=True, help="Path to the output folder where concatenated data will be saved."
    )
    parser.add_argument(
        "--ref_modality",
        type=str,
        required=True,
        help="Reference modality to which all other modalities will be resampled. Example: 'CT'",
    )
    parser.add_argument(
        "--modality-mapping",
        type=str,
        required=True,
        help="Comma-separated list of modalities to concatenate. Example: 'CT:_ct.nii.gz,PT:_pet.nii.gz'",
    )
    return parser


def concatenate(data: Dict[str, Any], ref_modality: str, output_folder: str) -> Any:
    modalities = list(data.keys())
    filenames = [data[modality] for modality in modalities]
    # Extract the suffix as the part after the last "_" or "-" in the filename
    base = filenames[0]
    if "_" in base and (base.rfind("_") > base.rfind("-")):
        suffix = "_" + base.split("_")[-1]
    elif "-" in base:
        suffix = "-" + base.split("-")[-1]
    else:
        suffix = ".nii.gz"
    load = Compose(
        [LoadImage(), EnsureChannelFirst()]
    )  # LoadImage will handle NIfTI files and EnsureChannelFirst ensures the channel dimension is first
    resample = SpatialResample(mode="bilinear")
    concatenate = ConcatItemsd(keys=list(modalities), name="image")
    formatter = NIFTINameFormatter(suffix=suffix)
    save = SaveImage(output_dir=output_folder, output_name_formatter=formatter, output_postfix="image", separate_folder=False)

    for modality, filename in zip(modalities, filenames):
        data[modality] = load(filename)
    for modality in modalities:
        if modality != ref_modality:
            try:
                source_affine_4x4 = define_affine_from_meta(data[modality].meta)
                data[modality].meta["affine"] = torch.Tensor(source_affine_4x4)
            except KeyError:
                source_affine_4x4 = data[modality].meta["affine"]

            try:
                target_affine_4x4 = define_affine_from_meta(data[ref_modality].meta)
            except KeyError:
                target_affine_4x4 = data[ref_modality].meta["affine"]
            if modality == "BOXES" or modality == "SEG":
                # Set all voxel values >0 to 100 for this modality
                img = data[modality]
                img_array = img[0] if img.shape[0] == 1 else img
                img_array = img_array.clone()  # Avoid modifying original
                img_array[img_array > 0] = 100
                # If EnsureChannelFirst has been used, repack as channel-first
                if img.shape[0] == 1:
                    img[0] = img_array
                else:
                    img = img_array
                data[modality] = img
       
            data[modality] = resample(data[modality], dst_affine=target_affine_4x4, spatial_size=data[ref_modality].shape[1:])
    data = concatenate(data)["image"]
    save(data)
    return filenames[0][: -len(suffix)] + "_image.nii.gz"


def main():
    data_folder = "/opt/data/Task101_BraTS-MET-2026/raw_splitted/imagesTs"
    prediction_folder = "/opt/models/Task101_BraTS-MET-2026/RetinaUNetV001_D3V001_3d/consolidated/test_predictions_nii"
    output_folder = "/opt/models/Task101_BraTS-MET-2026/RetinaUNetV001_D3V001_3d/consolidated/test_predictions_nii"
    data = {
        "T1C": str(next(Path(data_folder).glob("*_0000.nii.gz"))),
        "T1N": str(next(Path(data_folder).glob("*_0001.nii.gz"))),
        "T2F": str(next(Path(data_folder).glob("*_0002.nii.gz"))),
        "T2W": str(next(Path(data_folder).glob("*_0003.nii.gz"))),
        "BOXES": str(next(Path(prediction_folder).glob("*_boxes.nii.gz"))),
        "SEG": str(next(Path(prediction_folder).glob("*_seg.nii.gz"))),
   
    }
    concatenate(data, "T1C", output_folder)


if __name__ == "__main__":
    main()
