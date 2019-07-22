
# class-wise GAN-based semi-supervised learning


## Requirements

> python 2.7

> pytorch 0.3.0
 (To install pytorch, run `conda install pytorch=0.3.0 cuda80 -c soumith`.)

> argparse

> numpy 1.13.3

> matplotlib 2.2.2

> scikit-learn 0.19.1

> scipy 0.19.1

> PIL 5.1.0

## Run the Code

To reproduce our results
```
python multi_decoder_trainer.py -suffix test001 -max_epochs 8 -dataset imagenet10 -num_label 10 -ld 3000 -image_side 64 -gen_mode 34 -train_step 1
python multi_decoder_trainer.py -suffix test001 -max_epochs 30 -dataset imagenet10 -num_label 10 -ld 3000 -image_side 64 -gen_mode 34 -train_step 2 -step1_epo 8 -dis_channels 512 -con_coef 5e-1 -dgl_weight 1e-2 -uc -ef_weight 3e-1 -gf_weight 7e-1 -gl_weight 8e-1 -gr_weight 1e-1 -gc_weight 1e-3 -gn_weight 1e-5 -nei_margin 5e-2 -dg_ratio 5 -eg_ratio 1
```

To reproduce baseline results
```
python svhn_trainer.py
python cifar_trainer.py
```


## Results

Here is a comparison of different models using standard architectures without ensembles (4000 labels on CIFAR, 5000 labels on STL10, 20 labels on COIL20, 3000 labels on imagenet10):

Method | CIFAR (% accuracy) | STL10 (% accuracy) | COIL20 (% accuracy) | imagenet10 (% accuracy)
-- | -- | -- | -- | --
Baseline | **83.22** | **72.94** | **65.14** | **70.53**
Ours | **81.47** | **75.86** | **7577.6486** | **73.60**

