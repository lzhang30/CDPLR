# CDPLR
A description projection for our JAMIT presentation `Semi-Scribble MRI Image Segmentation via Confidence- and Distance-based Pseudo-label Refinement'

Download the dataset from the ACDC website and prepare the dataset following   [WSL4MIS](https://github.com/HiLab-git/WSL4MIS) 

It is important that the whole 150 cases of the ACDC dataset should be preprocessed.

Train the model with 8 labeled volumes:
```
cd code
python train_dist_unce.py --gpu 0 --labeled_ratio 8 --check 500 --early_stop 10000 --fold fold5 --num_classes 4 --root_path ../data/ACDC --exp ACDC_dist_unce --max_iterations 100000 --batch_size 16 &
python train_dist_unce.py --gpu 1 --labeled_ratio 8 --check 500 --early_stop 10000 --fold fold5 --num_classes 4 --root_path ../data/ACDC --exp ACDC_dist_unce --max_iterations 100000 --batch_size 16 &
python train_dist_unce.py --gpu 2 --labeled_ratio 8 --check 500 --early_stop 10000 --fold fold5 --num_classes 4 --root_path ../data/ACDC --exp ACDC_dist_unce --max_iterations 100000 --batch_size 16 &
python train_dist_unce.py --gpu 3 --labeled_ratio 8 --check 500 --early_stop 10000 --fold fold5 --num_classes 4 --root_path ../data/ACDC --exp ACDC_dist_unce --max_iterations 100000 --batch_size 16 &
python train_dist_unce.py --gpu 4 --labeled_ratio 8 --check 500 --early_stop 10000 --fold fold0 --num_classes 4 --root_path ../data/ACDC --exp ACDC_dist_unce --max_iterations 60000 --batch_size 16

python val_ours.py
```
The result of your experiment may be shown in `output_8.csv`

Have fun!
