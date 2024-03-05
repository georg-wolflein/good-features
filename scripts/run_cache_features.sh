#!/bin/bash
set -e

echo ====================
echo RUNNING: env/bin/python -m histaug.train.cache dataset=_tcga_brca augmentations@dataset.augmentations=simple_rotate settings.feature_extractor=owkin_teacher dataset.num_workers=16
echo ====================
env/bin/python -m histaug.train.cache dataset=_tcga_brca augmentations@dataset.augmentations=simple_rotate settings.feature_extractor=owkin_teacher dataset.num_workers=16

echo ====================
echo RUNNING: env/bin/python -m histaug.train.cache dataset=_tcga_brca augmentations@dataset.augmentations=all settings.feature_extractor=owkin_teacher dataset.num_workers=16
echo ====================
env/bin/python -m histaug.train.cache dataset=_tcga_brca augmentations@dataset.augmentations=all settings.feature_extractor=owkin_teacher dataset.num_workers=16

echo ====================
echo RUNNING: env/bin/python -m histaug.train.cache dataset=_tcga_brca augmentations@dataset.augmentations=simple_rotate settings.feature_extractor=mocov2 dataset.num_workers=16
echo ====================
env/bin/python -m histaug.train.cache dataset=_tcga_brca augmentations@dataset.augmentations=simple_rotate settings.feature_extractor=mocov2 dataset.num_workers=16

echo ====================
echo RUNNING: env/bin/python -m histaug.train.cache dataset=_tcga_brca augmentations@dataset.augmentations=all settings.feature_extractor=mocov2 dataset.num_workers=16
echo ====================
env/bin/python -m histaug.train.cache dataset=_tcga_brca augmentations@dataset.augmentations=all settings.feature_extractor=mocov2 dataset.num_workers=16

echo ====================
echo RUNNING: env/bin/python -m histaug.train.cache dataset=_tcga_crc augmentations@dataset.augmentations=simple_rotate settings.feature_extractor=owkin_teacher dataset.num_workers=16
echo ====================
env/bin/python -m histaug.train.cache dataset=_tcga_crc augmentations@dataset.augmentations=simple_rotate settings.feature_extractor=owkin_teacher dataset.num_workers=16

echo ====================
echo RUNNING: env/bin/python -m histaug.train.cache dataset=_tcga_crc augmentations@dataset.augmentations=all settings.feature_extractor=owkin_teacher dataset.num_workers=16
echo ====================
env/bin/python -m histaug.train.cache dataset=_tcga_crc augmentations@dataset.augmentations=all settings.feature_extractor=owkin_teacher dataset.num_workers=16

echo ====================
echo RUNNING: env/bin/python -m histaug.train.cache dataset=_tcga_crc augmentations@dataset.augmentations=simple_rotate settings.feature_extractor=mocov2 dataset.num_workers=16
echo ====================
env/bin/python -m histaug.train.cache dataset=_tcga_crc augmentations@dataset.augmentations=simple_rotate settings.feature_extractor=mocov2 dataset.num_workers=16

echo ====================
echo RUNNING: env/bin/python -m histaug.train.cache dataset=_tcga_crc augmentations@dataset.augmentations=all settings.feature_extractor=mocov2 dataset.num_workers=16
echo ====================
env/bin/python -m histaug.train.cache dataset=_tcga_crc augmentations@dataset.augmentations=all settings.feature_extractor=mocov2 dataset.num_workers=16

