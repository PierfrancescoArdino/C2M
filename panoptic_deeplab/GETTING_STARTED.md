## Getting Started with Panoptic-DeepLab

### Simple demo
Please download pre-trained model from [MODEL_ZOO](MODEL_ZOO.md), replace CONFIG_FILE with
corresponding config file of model you download, modify the input and output folders and then run

```bash
python tools/generate_segmentation.py --cfg configs/CONFIG_FILE \
    TEST.MODEL_FILE YOUR_DOWNLOAD_MODEL_FILE
```
