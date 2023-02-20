# Final Year Project - Self-supervised Transformers

Coursework created for the COM3001 Professional Project module during my Computer Science BSc at Surrey University. Completed Year 3 Summer 2022.

‘Self-supervised transformers for the downstream classification of diverse systemic conditions’, supervised by Professor H Lilian Tang.

I experimented with unsupervised and transfer learning with vision transformers, specifically the Swin Transformer for the diagnosis of systemic medical conditions using retinal images (fundus images). 

The data used for this project was sourced from largescale real-world public datasets, these were EyePACS and the UK biobank.

One of the models created during the project achieved equivalent performance to previous industrystandard research in the diabetic retinopathy grading task using a significantly smaller image size. 

This project also proved that the task of medical diagnosis from retinal images
can be done, and with fewer computational resources than previously thought, with
reasonable accuracy (to detect dementia, hypertension, and renal issues), as useful
knowledge can be transferred from domains with an abundance of data (such as ImageNet
or EyePACS) to domains with limited samples. 

Finally, the model’s image focus points
aligned with past theoretical research, showing that those areas can be potential research
areas to discover biomarkers for disease. These focus points were identified using class
activation mapping with the attention layers of the architecture. 

Processing for this project was done using Surrey’s HPC AI cluster scheduled using HTCondor.


--------------------------------------------------------------------------------------------------------------------------------------------

Code used throughout this project uses strings for paths to directories and file paths that will need to be replaced with the correct paths on your system.

Relevant excel files are provided, giving the logs and graphs for each of the experiments.

Contents:
All code, notebooks, requirements files, scripts, submit files and excel files that log the results of the experiments.

All models and training logs are stored in the /models/ folder for each experiment.

All processed UK biobank data is stored in the /BioBankData/ folder.

/DR DATA/ folder contains the DR fundus image data for the unsupervised training as well as the supervised set with a 20% train/val split.

/fundusImg/ folder contains all diagnosis fundus images used from the UK biobank and ODIR, including pure samples for supervised learning, for their prediction.


--------------------------------------------------------------------------------------------------------------------------------------------
Firstly, the fundus image cropping experiments were done in Kaggle notebooks as this didn't require downloading the whole DR dataset.
The full resolution DR train set was very large,>100GB.

If experimenting from scratch on the full DR dataset, the following steps are required:
1. Download the DR dataset from https://www.kaggle.com/competitions/diabetic-retinopathy-detection
2. Unzip the dataset

For any datasets downloaded from Kaggle, the following steps are required:
a Kaggle API key will be needed to download the DR dataset.
For Google Colab users, the Kaggle API key needs to be uploaded in the root dir "/content/"
Then run the following:
"""
! pip install kaggle
! mkdir ~/.kaggle
! cp kaggle.json ~/.kaggle/
! chmod 600 ~/.kaggle/kaggle.json
! mkdir competitionKaggle
! pip install --upgrade --force-reinstall --no-deps kaggle

!kaggle datasets download -p path/where/you/want/it/Downloaded -d datasetUploaderUserName/DatasetName

!unzip path/where/you/want/it/Downloaded/DatasetName.zip -d /path/where/you/want/it/extracted/
!rm path/where/you/want/it/Downloaded/DatasetName.zip
"""

The Kaggle image cropping code can be found in: "CodeFiles\Fundus image pre-process\FundusCroppingResizeExperimetns.ipynb"

For the Kaggle experiments:
libs used where
1. OS
2. Matplotlib
3. cv2
4. numpy
5. pylab
6. pandas
7. glob
8. tqdm
9. Math
10. PIL
11. pandas

A requirements file will be provided for the Kaggle experiments for specific versioning. However, only the libs listed above are required.
"CodeFiles\Fundus image pre-process\kaggleRequirements.txt"

--------------------------------------------------------------------------------------------------------------------------------------------

# SWIN MOBY 224 COLAB

For the unsupervised learning using Swin MOBY, this is done for image size 224px on google colab
The training was done with GPUs with 16GB of memory, so changes to the batch size will be required for successful training with different hardware.

The code for this can be found in: "SWIN_SSL_MOBY_Colab.ipynb"

The full DR train set (with all data in the train folder) was used for this and any other unsupervised learning experiments. - This will be provided in full in the data directory
"DR DATA\DR unsupervised fundus images" - this will need to be unzipped.

For loading pre-trained models using this code, the following steps are required:
#https://github.com/microsoft/Swin-Transformer/pull/184/commits/821c0161079acb11e36ae7d5ac03b1a35d4aa205
To change the pre-trained model loading code to accept MOBY and ImageNet models.

Changing paths will be required for the software to work with the directories and paths to data.
I recommend using Colab to run this however, I will provide a full requirements file - although it will contain a lot of unnecessary libs.

The models and logs can be found in the "models\300ep224Colab" directory - this will need to be unzipped.

------------------------------------------------
# SWIN MOBY 448, 672 CONDOR

For the unsupervised learning using Swin MOBY, this is done for image sizes 672px and 448px
This was done on Condor. The anaconda environment specification files are provided in the "code/CodeFiles/condor" directory.
The environment with full compatibility with 3090 GPUs is "transformer-ssl-3090.yml" or the txt version obtained using pip freeze
"CodeFiles\condor\transformer-ssl-3090.yml"
"CodeFiles\condor\transformer-ssl-3090.txt"
"CodeFiles\condor\transformer-ssl.yml"
"CodeFiles\condor\transformer-ssl.txt"

The data directory is "DR DATA\DR unsupervised fundus images" - this will need to be unzipped.

"RStartPre.sh" and its submit file are used to initialize from a pre-trained model and start training. They are not configured to resume training from the latest checkpoint if run again.

"Real.sh" and its submit file are used to train the model, configured to resume from the latest checkpoint if avalible.
Both will save checkpoints to the specified output directory.

They use two versions of the code, "CodeFiles\condor\condor-examples\Mine\Transformer-SSL-3090-Pre", and "CodeFiles\condor\condor-examples\Mine\Transformer-SSL-3090"
"CodeFiles\condor\condor-examples\Mine\Transformer-SSL" is a native version without added support for later GPU models and can be run on heron machines.

The models and logs can be found in the "models\300ep672" and "models\1ep448" directories.

------------------------------------------------
# DR eval training

For model evaluation on the DR test set using Swin
The code from "SwinSupervisedColab.ipynb" should include a download of some pre-trained models from Kaggle - the Kaggle setup as before needs to be done for this to work.
The same Google Colab setup was used as before, and so the requirements will be similar
another requirements file will be provided just in case.

An earlier commit was used as there was no official release to base the download from Git, and the repo is constantly updated with changes that can potentially break the code.

Again the requirements will most likely contain a lot of unnecessary libs.

The data for this evaluation is provided under the "DR DATA\DR supervised fundus images 20split" directory.
This same split can be recreated using the same code in "SWIN_SSL_MOBY_Colab.ipynb" - for downloading, converting and then spiting the data into Train and Val folders

The code for this part will take the pre-trained models and train for a few epochs on each.

The models and logs can be found in the "models\DREval" directory - these will need to be unzipped.

------------------------------------------------
# Log metrics extraction

For extracting the metrics from the log files, the code is provided in "Log extraction.ipynb"

These scripts were created to extract the relevant metrics from the Swin MoBY unsupervised as well as the Swin supervised log files.

This was done using the default Google Colab setup, and so the requirements will be similar - no extra installs were required
The libraries required are:
1. pprint
2. pandas


------------------------------------------------
# Additional metrics extraction

To extract additional metrics from a Swin supervised models using validation data - this is done in "CodeFiles/SwinTorchMetrics.ipynb"
The same data from DR eval training can be used in the "DR DATA\DR supervised fundus images 20split" directory.
Although the method to create from scratch is also provided - as the validation selection code is deterministic and can be reproduced.

The version provided was used in Colab with the default setup, so the requirements will be similar - however, some libraries will be installed in the script.

Depending on the model and images used, the batch size will need to be adjusted to work with other setups
comments are provided to explain the batch size for different image sizes using a GPU with 16GB of VRAM

This code was also used for the disease classification task with a different class number of 2 but still using the multi-class classification metrics
as described in the report. An edited version is provided in the file: "CodeFiles/SwinTorchMetricsDiagnosis.ipynb"

--------------------------------------------------------------------------------------------------------------------------------------------
# SWIN disease classification

For the supervised learning with Swin done for the different disease classification, this uses the same "transformer-ssl-3090" anaconda environment.
This uses the default Swin implementation found under "CodeFiles\condor\condor-examples\Mine\Swin-Transformer".

The version included is slightly older, using a specific commit as the repo is updated regularly, as mentioned before, and there is no specific release.

The launch scripts and submit files are provided in the "CodeFiles\condor\condor-examples\Mine\biobankjobs".

The data used for this training is provided in the "fundusImg" directory. Each diagnosis has a folder containing its train and val subdirectories.
The subdirectories contain the positive samples (1) and negative samples (0) (pure samples) in a train/val split of 20%. 

Additional pure (without any diagnosis) fundus image samples are also provided in the same folder under the subdirectory "PureSamples".

The models, logs and confusion matrices for this can be found in "models\diagnosis" folder - along with the models used for initialization.


-------------------------------------------------
# data processing:

Code can be found in:
"CodeFiles\Data processing\Uk_BioBank_data.ipynb"
"CodeFiles\Data processing\FileScripts.ipynb"
"CodeFiles\Data processing\FYPFiles.ipynb"

These contain code for:
-processing the UK biobank data from raw
-format to CSV
-filtering samples
-finding diagnosis columns
-creating an index of all samples for each disease in ICD-10
-creating an index of all samples without a diagnosis
-cropping samples
-moving samples to their respective folders
-utility scripts for cleaning diagnosis datasets
-cropping images to remove the black border and resizing to a more manageable size

additional code is provided for pulling pure and hypertension samples from the ODIR-2019 dataset
-and moving a proportional number of pure samples to the negative folder for each disease
-and splitting each dataset into train/val

Some of this code might be duplicated. However, this will mostly be the small scripts to move files around.

These were used with a local environment - a requirements file will be provided with anaconda under "CodeFiles/LocalENV.yaml"


--------------------------------------------------------------------------------------------------------------------------------------------
# SWIN CAM

This was done using a local environment - with the same requirements file as in data processing.

The code used for this project is provided in:
"CodeFiles\Swin CAM\SwinCAM.ipynb"

older experimental code that wasn't used can be found in:
"CodeFiles\Swin CAM\SwinVisualisation.ipynb"

--------------------------------------------------------------------------------------------------------------------------------------------
