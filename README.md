# CDPLR
A description projection for our JAMIT presentation `Semi-Scribble MRI Image Segmentation via Confidence- and Distance-based Pseudo-label Refinement'

Download the dataset from the ACDC website and prepare the dataset following   [WSL4MIS](https://github.com/HiLab-git/WSL4MIS) 

It is important that the whole 150 cases of the ACDC dataset should be preprocessed.

Train the model with 8 labeled volumes:
```
cd code
python train_dist_unce.py --gpu 0 --labeled_ratio 8 --check 500 --early_stop 10000 --fold fold5 --num_classes 4 --root_path ../data/ACDC --exp ACDC_dist_unce --max_iterations 60000 --batch_size 16 &
python train_dist_unce.py --gpu 1 --labeled_ratio 8 --check 500 --early_stop 10000 --fold fold5 --num_classes 4 --root_path ../data/ACDC --exp ACDC_dist_unce --max_iterations 60000 --batch_size 16 &
python train_dist_unce.py --gpu 2 --labeled_ratio 8 --check 500 --early_stop 10000 --fold fold5 --num_classes 4 --root_path ../data/ACDC --exp ACDC_dist_unce --max_iterations 60000 --batch_size 16 &
python train_dist_unce.py --gpu 3 --labeled_ratio 8 --check 500 --early_stop 10000 --fold fold5 --num_classes 4 --root_path ../data/ACDC --exp ACDC_dist_unce --max_iterations 60000 --batch_size 16 &
python train_dist_unce.py --gpu 4 --labeled_ratio 8 --check 500 --early_stop 10000 --fold fold0 --num_classes 4 --root_path ../data/ACDC --exp ACDC_dist_unce --max_iterations 60000 --batch_size 16

python val_ours.py
```
The result of your experiment should be shown in `output_8.csv`

Have fun!

**Acknowledgement**

A part of our code is from [WSL4MIS](https://github.com/HiLab-git/WSL4MIS) and [DMPLS](https://arxiv.org/abs/2203.02106)

I am grateful to **Dr. Zhengzhou** for the technical support.
