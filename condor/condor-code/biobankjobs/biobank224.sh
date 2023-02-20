eval "$('/user/HS126/gb00538/anaconda3/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
conda activate transformer-ssl-3090 
#-112
cd /user/HS126/gb00538/condor-examples/Mine/Swin-Transformer

export batch_size="64" #32 #16 #8 #4 ish (grad overflows)
export image_size="256"  #"224" #[224, 448, 672], 896
export TRAIN_EPOCHS="50" #300
export SWIN_WINDOW_SIZE="7" #7, 14, 21
#export TRAIN_WARMUP_EPOCHS="1"
export NUM_CLASSES="2"
export NUM_GPU="1"
#export MasterOrSlave="1"
export SAVE_FREQ="1"

#"O0", "O1", "O2
export ampOptLevel="O1" #default 01

python -m torch.distributed.launch --nproc_per_node $NUM_GPU --master_port 12345  "/user/HS126/gb00538/condor-examples/Mine/Swin-Transformer/main.py" \
--cfg configs/swin_tiny_patch4_window7_224.yaml \
--data-path '/vol/research/bioAI-GB2022/fundusImg/Ch_IX_Diseases_of_Circulatory_System/' \
--batch-size $batch_size \
--output '/vol/research/bioAI-GB2022/models/Ch_IX_Diseases_of_Circulatory_System/' \
--amp-opt-level $ampOptLevel \
--opts TRAIN.EPOCHS $TRAIN_EPOCHS MODEL.NUM_CLASSES $NUM_CLASSES DATA.IMG_SIZE $image_size MODEL.SWIN.WINDOW_SIZE $SWIN_WINDOW_SIZE \
SAVE_FREQ $SAVE_FREQ --tag "swin224Window7_Ch_IX_Diseases_of_Circulatory_System"



#python -m torch.distributed.launch --nproc_per_node $NUM_GPU --master_port 12345  "/content/Swin-Transformer/main.py" \
#--cfg configs/swin_tiny_patch4_window7_224.yaml \
#--data-path '/content/competitionKaggle/' --batch-size $batch_size  \
#--output '/content/drive/MyDrive/Colab Notebooks/FYP/MOBY SSL SWIN/' \
#--opts TRAIN.EPOCHS $TRAIN_EPOCHS TRAIN.WARMUP_EPOCHS $TRAIN_WARMUP_EPOCHS MODEL.NUM_CLASSES $NUM_CLASSES DATA.IMG_SIZE $image_size \
#MODEL.SWIN.WINDOW_SIZE $SWIN_WINDOW_SIZE 


#--output '/vol/research/Neurocomp/DR-Fundus-images/FullSet/' \
#--output '/vol/research/Neurocomp/DR-Fundus-images/FullSet/2xLargerbatch/' \
#--output '/vol/research/Neurocomp/DR-Fundus-images/FullSet/LargerBatchRun/' \

