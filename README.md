## ELAN
Codes for "Efficient Long-Range Attention Network for Image Super-resolution", [arxiv link](https://arxiv.org/abs/2203.06697).

> **Efficient Long-Range Attention Network for Image Super-resolution** <br>
> [Xindong Zhang](https://github.com/xindongzhang), [Hui Zeng](https://huizeng.github.io/), [Shi Guo](https://scholar.google.com.hk/citations?user=5hsEmuQAAAAJ&hl=zh-CN), and [Lei Zhang](https://www4.comp.polyu.edu.hk/~cslzhang/). <br>
> In ECCV 2022.


## Training
```
cd ELAN
python train.py --config ./configs/elan_light_x4.yml
```

## Testing
```
cd ELAN
python test.py --config ./configs/elan_light_x2.yml --model_path ./model.pt --input_image_path ./input.jpg --output_image_path ./output.jpg
```

