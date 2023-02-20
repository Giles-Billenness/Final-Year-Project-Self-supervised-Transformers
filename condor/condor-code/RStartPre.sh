eval "$('/user/HS126/gb00538/anaconda3/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
conda activate transformer-ssl-3090
cd /user/HS126/gb00538/condor-examples/Mine/Transformer-SSL-3090-Pre

export batch_size="8" #"16" #32 #16 #8 #4 ish (grad overflows)
export image_size="672" #224, 448, 672, 896
export TRAIN_EPOCHS="300" #300
export SWIN_WINDOW_SIZE="21" #7, 14, 21
#export TRAIN_WARMUP_EPOCHS="1"
export NUM_CLASSES="5"
export NUM_GPU="4"
#export MasterOrSlave="1"
export SAVE_FREQ="1"

python -m torch.distributed.launch --nproc_per_node $NUM_GPU --master_port 12345  "/user/HS126/gb00538/condor-examples/Mine/Transformer-SSL-3090-Pre/moby_main.py" \
--cfg configs/moby_swin_tiny.yaml --data-path '/vol/research/Neurocomp/DR-Fundus-images/FullSet/' --batch-size $batch_size \
--output '/vol/research/Neurocomp/DR-Fundus-images/FullSet/LargerBatchRun/' \
--opts TRAIN.EPOCHS $TRAIN_EPOCHS MODEL.NUM_CLASSES $NUM_CLASSES DATA.IMG_SIZE $image_size MODEL.SWIN.WINDOW_SIZE $SWIN_WINDOW_SIZE \
SAVE_FREQ $SAVE_FREQ \
--pretrained "/vol/research/Neurocomp/DR-Fundus-images/FullSet/ckpt_672px_epoch_45.pth" \



#--output '/vol/research/Neurocomp/DR-Fundus-images/FullSet/2xLargerbatch/' \
#--output '/vol/research/Neurocomp/DR-Fundus-images/FullSet/LargerBatchRun/' \

#--pretrained "/vol/research/Neurocomp/DR-Fundus-images/FullSet/ckpt_672px_epoch_45.pth" \
#--pretrained "/vol/research/Neurocomp/DR-Fundus-images/FullSet/ckpt_672px_epoch_40.pth" \ (BUGGED)
#--pretrained "/vol/research/Neurocomp/DR-Fundus-images/FullSet/ckpt_672px_epoch_45.pth" \
