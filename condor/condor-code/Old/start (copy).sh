eval "$('/user/HS126/gb00538/anaconda3/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
conda activate transformer-ssl
cd /vol/vssp/cvpnobackup/scratch_4weeks/qh00006/punc_interspeech
python train.py

