eval "$('/user/HS126/gb00538/anaconda3/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
conda activate transformer-ssl-3090 
#-112
cd /user/HS126/gb00538/condor-examples/Mine/Transformer-SSL-3090

export batch_size="16" #32 #16 #8 #4 ish (grad overflows)
export image_size="448" #[224, 448, 672], 896
export TRAIN_EPOCHS="300" #300
export SWIN_WINDOW_SIZE="14" #7, 14, 21
#export TRAIN_WARMUP_EPOCHS="1"
export NUM_CLASSES="5"
export NUM_GPU="4"
#export MasterOrSlave="1"
export SAVE_FREQ="1"

#"O0", "O1", "O2
export ampOptLevel="O1" #default 01

python -m torch.distributed.launch --nproc_per_node $NUM_GPU --master_port 12345  "/user/HS126/gb00538/condor-examples/Mine/Transformer-SSL-3090/moby_main.py" \
--cfg configs/moby_swin_tiny.yaml --data-path '/vol/research/Neurocomp/DR-Fundus-images/FullSet/' --batch-size $batch_size \
--output '/vol/research/Neurocomp/DR-Fundus-images/FullSet/2xLargerbatch/' \
--amp-opt-level $ampOptLevel \
--opts TRAIN.EPOCHS $TRAIN_EPOCHS MODEL.NUM_CLASSES $NUM_CLASSES DATA.IMG_SIZE $image_size MODEL.SWIN.WINDOW_SIZE $SWIN_WINDOW_SIZE \
SAVE_FREQ $SAVE_FREQ


#--output '/vol/research/Neurocomp/DR-Fundus-images/FullSet/' \
#--output '/vol/research/Neurocomp/DR-Fundus-images/FullSet/2xLargerbatch/' \
#--output '/vol/research/Neurocomp/DR-Fundus-images/FullSet/LargerBatchRun/' \

