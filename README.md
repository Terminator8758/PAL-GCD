# Prior-constrained Association Learning for Fine-grained Generalized Category Discovery

This repo contains the implementation code of our paper: [Prior-constrained Association Learning for Fine-grained Generalized Category Discovery](https://arxiv.org/abs/2502.09501).

![teaser](figure/framework.pdf)
The proposed method PAL-GCD is primarily focused on non-parametric classification through prototypical contrastive learning and prior-constrained data association. Additionally, it also provides the combination of parametric and non-parametric classification by which a higher performance can be obtained.


## Running

### Dependencies

```
pip install -r requirements.txt
```

### Config

Set paths to datasets and desired log directories in ```config.py```


### Datasets

We use fine-grained benchmarks in this paper, including:

* [The Semantic Shift Benchmark (SSB)](https://github.com/sgvaze/osr_closed_set_all_you_need#ssb) and [Herbarium19](https://www.kaggle.com/c/herbarium-2019-fgvc6)

We also use generic object recognition datasets, including:

* [CIFAR-100](https://pytorch.org/vision/stable/datasets.html) and [ImageNet-100](https://image-net.org/download.php)


### Scripts

**Train the model**:
- Train with only non-parametric classifier:
```
sh run_${DATASET_NAME}_stage1.sh
```

- Train with joint parametric and non-parametric classifier:
```
sh run_${DATASET_NAME}_stage1_and_stage2.sh
```



## Citation

If you find this repo useful for your research, please consider citing our paper:

```
@inproceedings{wang2025palGCD,
  title={Prior-constrained Association Learning for Fine-grained Generalized Category Discovery},
  author={Menglin Wang and Zhun Zhong and Xiaojin Gong},
  booktitle={AAAI Conference on Artificial Intelligence},
  year={2025}
}
```

## Acknowledgements

The codebase is largely built on this repo: https://github.com/CVMI-Lab/SimGCD. Thanks to the authors for their method implementation.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
