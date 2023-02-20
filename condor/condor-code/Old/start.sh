eval "$('/user/HS126/gb00538/anaconda3/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
conda activate transformer-ssl
cd /user/HS126/gb00538/condor-examples/Mine/Transformer-SSL

export batch_size="8" #32 #16 #8 #4 ish (grad overflows)
export image_size="224" #*2 #*3 #*4
export TRAIN_EPOCHS="5" #300
export SWIN_WINDOW_SIZE="21"
export TRAIN_WARMUP_EPOCHS="1"
export NUM_CLASSES="5"
export NUM_GPU="1"
#export MasterOrSlave="1"

python -m torch.distributed.launch --nproc_per_node $NUM_GPU --master_port 12345  "/user/HS126/gb00538/condor-examples/Mine/Transformer-SSL/moby_main.py" \
--cfg configs/moby_swin_tiny.yaml --data-path '/vol/research/Neurocomp/DR-Fundus-images/FullSet/' --batch-size $batch_size \
--output '/vol/research/Neurocomp/DR-Fundus-images/FullSet/' \
--opts TRAIN.EPOCHS $TRAIN_EPOCHS TRAIN.WARMUP_EPOCHS $TRAIN_WARMUP_EPOCHS MODEL.NUM_CLASSES $NUM_CLASSES DATA.IMG_SIZE $image_size 

: '
python -m torch.distributed.launch --nproc_per_node $NUM_GPU --master_port 12345  "/user/HS126/gb00538/condor-examples/Mine/Transformer-SSL/moby_main.py" \
--local_rank $MasterOrSlave \
--cfg configs/moby_swin_tiny.yaml --data-path '/vol/research/Neurocomp/DR-Fundus-images/Demo/' --batch-size $batch_size \
--output '/vol/research/Neurocomp/DR-Fundus-images/Demo/' \
--opts TRAIN.EPOCHS $TRAIN_EPOCHS TRAIN.WARMUP_EPOCHS $TRAIN_WARMUP_EPOCHS MODEL.NUM_CLASSES $NUM_CLASSES DATA.IMG_SIZE $image_size LOCAL_RANK $MasterOrSlave
'


