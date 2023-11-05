#!/bin/bash
set -e

echo ====================
echo RUNNING: env/bin/python -m histaug.train.cache dataset=_tcga_crc augmentations@dataset.augmentations=simple_rotate settings.feature_extractor=ctranspath dataset.num_workers=16 dataset.cache_dir='/raid/histaug/cache/TCGA-CRC/${settings.feature_extractor}/${.augmentations.name}'
echo ====================
env/bin/python -m histaug.train.cache dataset=_tcga_crc augmentations@dataset.augmentations=simple_rotate settings.feature_extractor=ctranspath dataset.num_workers=16 dataset.cache_dir='/raid/histaug/cache/TCGA-CRC/${settings.feature_extractor}/${.augmentations.name}'

echo ====================
echo RUNNING: env/bin/python -m histaug.train.cache dataset=_tcga_crc augmentations@dataset.augmentations=simple_rotate settings.feature_extractor=owkin dataset.num_workers=16 dataset.cache_dir='/raid/histaug/cache/TCGA-CRC/${settings.feature_extractor}/${.augmentations.name}'
echo ====================
env/bin/python -m histaug.train.cache dataset=_tcga_crc augmentations@dataset.augmentations=simple_rotate settings.feature_extractor=owkin dataset.num_workers=16 dataset.cache_dir='/raid/histaug/cache/TCGA-CRC/${settings.feature_extractor}/${.augmentations.name}'

echo ====================
echo RUNNING: env/bin/python -m histaug.train.cache dataset=_tcga_crc augmentations@dataset.augmentations=simple_rotate settings.feature_extractor=swin dataset.num_workers=16 dataset.cache_dir='/raid/histaug/cache/TCGA-CRC/${settings.feature_extractor}/${.augmentations.name}'
echo ====================
env/bin/python -m histaug.train.cache dataset=_tcga_crc augmentations@dataset.augmentations=simple_rotate settings.feature_extractor=swin dataset.num_workers=16 dataset.cache_dir='/raid/histaug/cache/TCGA-CRC/${settings.feature_extractor}/${.augmentations.name}'

echo ====================
echo RUNNING: env/bin/python -m histaug.train.cache dataset=_tcga_crc augmentations@dataset.augmentations=simple_rotate settings.feature_extractor=vit dataset.num_workers=16 dataset.cache_dir='/raid/histaug/cache/TCGA-CRC/${settings.feature_extractor}/${.augmentations.name}'
echo ====================
env/bin/python -m histaug.train.cache dataset=_tcga_crc augmentations@dataset.augmentations=simple_rotate settings.feature_extractor=vit dataset.num_workers=16 dataset.cache_dir='/raid/histaug/cache/TCGA-CRC/${settings.feature_extractor}/${.augmentations.name}'

echo ====================
echo RUNNING: env/bin/python -m histaug.train.cache dataset=_tcga_crc augmentations@dataset.augmentations=simple_rotate settings.feature_extractor=retccl dataset.num_workers=16 dataset.cache_dir='/raid/histaug/cache/TCGA-CRC/${settings.feature_extractor}/${.augmentations.name}'
echo ====================
env/bin/python -m histaug.train.cache dataset=_tcga_crc augmentations@dataset.augmentations=simple_rotate settings.feature_extractor=retccl dataset.num_workers=16 dataset.cache_dir='/raid/histaug/cache/TCGA-CRC/${settings.feature_extractor}/${.augmentations.name}'

echo ====================
echo RUNNING: env/bin/python -m histaug.train.cache dataset=_tcga_crc augmentations@dataset.augmentations=simple_rotate settings.feature_extractor=resnet50 dataset.num_workers=16 dataset.cache_dir='/raid/histaug/cache/TCGA-CRC/${settings.feature_extractor}/${.augmentations.name}'
echo ====================
env/bin/python -m histaug.train.cache dataset=_tcga_crc augmentations@dataset.augmentations=simple_rotate settings.feature_extractor=resnet50 dataset.num_workers=16 dataset.cache_dir='/raid/histaug/cache/TCGA-CRC/${settings.feature_extractor}/${.augmentations.name}'

echo ====================
echo RUNNING: env/bin/python -m histaug.train.cache dataset=_tcga_crc augmentations@dataset.augmentations=simple_rotate settings.feature_extractor=bt dataset.num_workers=16 dataset.cache_dir='/raid/histaug/cache/TCGA-CRC/${settings.feature_extractor}/${.augmentations.name}'
echo ====================
env/bin/python -m histaug.train.cache dataset=_tcga_crc augmentations@dataset.augmentations=simple_rotate settings.feature_extractor=bt dataset.num_workers=16 dataset.cache_dir='/raid/histaug/cache/TCGA-CRC/${settings.feature_extractor}/${.augmentations.name}'

echo ====================
echo RUNNING: env/bin/python -m histaug.train.cache dataset=_tcga_crc augmentations@dataset.augmentations=simple_rotate settings.feature_extractor=swav dataset.num_workers=16 dataset.cache_dir='/raid/histaug/cache/TCGA-CRC/${settings.feature_extractor}/${.augmentations.name}'
echo ====================
env/bin/python -m histaug.train.cache dataset=_tcga_crc augmentations@dataset.augmentations=simple_rotate settings.feature_extractor=swav dataset.num_workers=16 dataset.cache_dir='/raid/histaug/cache/TCGA-CRC/${settings.feature_extractor}/${.augmentations.name}'

echo ====================
echo RUNNING: env/bin/python -m histaug.train.cache dataset=_tcga_crc augmentations@dataset.augmentations=simple_rotate settings.feature_extractor=dino_p16 dataset.num_workers=16 dataset.cache_dir='/raid/histaug/cache/TCGA-CRC/${settings.feature_extractor}/${.augmentations.name}'
echo ====================
env/bin/python -m histaug.train.cache dataset=_tcga_crc augmentations@dataset.augmentations=simple_rotate settings.feature_extractor=dino_p16 dataset.num_workers=16 dataset.cache_dir='/raid/histaug/cache/TCGA-CRC/${settings.feature_extractor}/${.augmentations.name}'

