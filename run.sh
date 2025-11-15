python train_classifier_ssFT.py --data_name CIFAR100 --model_name resnet18 --d_mode new --device cuda:1  --cycles 500 --switch_epoch_pred 500  --control_name "1003_sup-ft_100_0.1_iid_5-5_0.07_1" --full_sup 1 --var_lr 0.03 --resume_mode 1 --train_pass 0 --register_hook_BN 1
python train_classifier_ssFT.py --data_name CIFAR100 --model_name resnet18 --d_mode new --device cuda:1 --cycles 500 --switch_epoch_pred 500  --control_name "10003_sup-ft_100_0.1_iid_5-5_0.07_1" --full_sup 1 --with_BN 0 --var_lr 0.03 --resume_mode 1 --train_pass 0 --register_hook_BN 1

/media/cds/DATA2/Yeswanth/SemiFL_unsup
python train_classifier_ssDA_target.py --domain_s 'art' --unsup_doms 'product'  --model_name resnet50 --d_mode new --device cuda:0  --var_lr 0.001  --scheduler_name  'ExponentialLR' --cycles 40  --control_name "1010014_sup-ft-fix_1_1_iid_5-5_0.07_1" --resume_mode 1  --init_seed 2020    --avg_cent 0 --gamma 1 --run_shot 0  --par 1 --tag_ "2020_art_0.01_resnet50_2223_sup-ft-fix"  --client_test 1

 python train_classifier_ssDA_target.py --domain_s 'art' --unsup_doms 'product'  --model_name resnet50 --d_mode new --device cuda:0  --var_lr 0.001  --scheduler_name  'ExponentialLR' --cycles 40  --control_name "1010012_sup-ft-fix_1_1_iid_5-5_0.07_1" --resume_mode 1  --init_seed 2020    --avg_cent 0 --gamma 1 --run_shot 0  --par 1 --tag_ "2020_art_0.01_resnet50_2223_sup-ft-fix"  --client_test 1




 python train_classifier_ssDA_targetAAAI.py --data_name 'DomainNetS'  --data_name_unsup 'DomainNetS'  --domain_s 'quickdraw' --unsup_doms 'clipart-painting-real-sketch-infograph'  --model_name VITs  --d_mode new --device cuda:1  --var_lr 0.03  --scheduler_name  'CosineAnnealingLR' --cycles 50  --control_name "04_sup-ft-fix_15_0.3_iid_5-5_0.07_1" --resume_mode  1  --init_seed 2020 --run_shot 1 --run_hcld 0 --par 1 --tag_ "2020_product_0.3_VITs_1_sup-ft-fix"  --client_test 1 --pick 'checkpoint'       --backbone_arch vit-small --add_fix 1   --lambda 1  --adpt_thr 1  --client_test 1 --g_lambda 0.1 --global_reg 1 --threshold 0.4