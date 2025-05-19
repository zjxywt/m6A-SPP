# m6A-SPP
This repository contains the implementation of 'DNABERT: a pre-trained Bidirectional Encoder Representations from Transformers model tailored for DNA-language within the genome'. Within this package, we offer resources such as the source code of the DNABERT model, usage examples, pre-trained models, and fine-tuned models. As this package is still in the process of development, more features will be incorporated incrementally. The training of DNABERT involves general pre-training and task-specific fine-tuning. As part of our project's contribution, we have made the pre-trained models available in this repository.

Citation
If DNABERT has been employed in your research, please cite the following publication:
Yanrong Ji, Zhihan Zhou, Han Liu, Ramana V Davuluri, DNABERT: pre-trained Bidirectional Encoder Representations from Transformers model for DNA-language in genome, Bioinformatics, 2021;, btab083, https://doi.org/10.1093/bioinformatics/btab083

Environment Configuration
We suggest creating a Python virtual environment using Anaconda. Additionally, ensure that you have at least one NVIDIA GPU with the Linux x86_64 Driver Version equal to or higher than 410.48 (compatible with CUDA 10.0). If you are using a GPU with different specifications or memory sizes, you may need to adjust the batch size accordingly.

1.1 Create and activate a new virtual environment
conda create -n dnabert python=3.6
conda activate dnabert
1.2 Install the package and other requirements
(Required)

conda install pytorch torchvision cudatoolkit=10.0 -c pytorch

git clone https://github.com/jerryji1993/DNABERT
cd DNABERT
python3 -m pip install --editable .
cd examples
python3 -m pip install -r requirements.txt

2. Pre-train (Skip this section if you fine-tune on pre-trained models)
2.1 Data processing
If you are trying to pre-train DNABERT with your own data, please process you data into the same format. Note that the sequences are in kmer format, so you will need to convert your sequences into that. 

In the following example, we use DNABERT with kmer=3 as example.

2.2 Model Training

python run_pretrain.py \
    --output_dir $OUTPUT_PATH \
    --model_type=dna \
    --tokenizer_name=dna$KMER \
    --config_name=$SOURCE/src/transformers/dnabert-config/bert-config-$KMER/config.json \
    --do_train \
    --train_data_file=$TRAIN_FILE \
    --do_eval \
    --eval_data_file=$TEST_FILE \
    --mlm \
    --gradient_accumulation_steps 25 \
    --per_gpu_train_batch_size 10 \
    --per_gpu_eval_batch_size 6 \
    --save_steps 500 \
    --save_total_limit 20 \
    --max_steps 200000 \
    --evaluate_during_training \
    --logging_steps 500 \
    --line_by_line \
    --learning_rate 4e-4 \
    --block_size 512 \
    --adam_epsilon 1e-6 \
    --weight_decay 0.01 \
    --beta1 0.9 \
    --beta2 0.98 \
    --mlm_probability 0.025 \
    --warmup_steps 10000 \
    --overwrite_output_dir \
    --n_process 24

3. Fine-tune (Skip this section if you use fine-tuned model)
3.1 Data processing
Please see the template data at /example/ft/. If you are trying to fine-tune DNABERT with your own data, please process you data into the same format as it. Note that the sequences are in kmer format, so you will need to convert your sequences into that. 

3.2 Download pre-trained DNABERT
DNABERT3
DNABERT4
DNABERT5
DNABERT6
Download the pre-trained model in to a directory. (If you would like to replicate the following examples, please download DNABERT 3). Then unzip the package by running:
unzip 6-new-12w-0.zip
Fine-tuned Model

3.3 Fine-tune with pre-trained model
In the following example, we use DNABERT with kmer=3 as example. We use prom-core, a 2-class classification task as example.

cd examples

export KMER=6
export MODEL_PATH=PATH_TO_THE_PRETRAINED_MODEL
export DATA_PATH=sample/ft/$KMER
export OUTPUT_PATH=./ft/$KMER

python run_finetune.py \
    --model_type dna \
    --tokenizer_name=dna$KMER \
    --model_name_or_path $MODEL_PATH \
    --task_name dnaprom \
    --do_train \
    --do_eval \
    --data_dir $DATA_PATH \
    --max_seq_length 100 \
    --per_gpu_eval_batch_size=32   \
    --per_gpu_train_batch_size=32   \
    --learning_rate 2e-4 \
    --num_train_epochs 5.0 \
    --output_dir $OUTPUT_PATH \
    --evaluate_during_training \
    --logging_steps 100 \
    --save_steps 4000 \
    --warmup_percent 0.1 \
    --hidden_dropout_prob 0.1 \
    --overwrite_output \
    --weight_decay 0.01 \
    --n_process 8

If you use the fine-tuned model instead of fine-tuning a model by your self, please download the fine-tuned and put it under examples/ft/3.

Fine-tuned Model

4. Prediction
After the model is fine-tuned, we can get predictions by running

export KMER=3
export MODEL_PATH=./ft/$KMER
export DATA_PATH=sample_data/ft/$KMER
export PREDICTION_PATH=./result/$KMER

python run_finetune.py \
    --model_type dna \
    --tokenizer_name=dna$KMER \
    --model_name_or_path $MODEL_PATH \
    --task_name dnaprom \
    --do_predict \
    --data_dir $DATA_PATH  \
    --max_seq_length 75 \
    --per_gpu_pred_batch_size=128   \
    --output_dir $MODEL_PATH \
    --predict_dir $PREDICTION_PATH \
    --n_process 48
With the above command, the fine-tuned DNABERT model will be loaded from MODEL_PATH , and makes prediction on the dev.tsv file that saved in DATA_PATH and save the prediction result at PREDICTION_PATH.

