# [사전학습]
accelerate launch --multi_gpu --num_processes=4 pre_train.py
# [파인튜닝]
accelerate launch --multi_gpu --num_processes=4 fine_train.py

# [평가데이터생성] ->[id,test_data캡션, test_data캡션으로 만든 MIDI, test_data_midi]
## 1. base REMI
python src/eval/prepare_test_captions_and_generate.py \ 
--captions_jsonl datasets/captions/captions_gukak_f.jsonl \
--tokenizer_vocab datasets/artifacts/vocab_remi.pkl \ 
--model_ckpt model/fine_output/epoch_300/kot2m_model.bin \
--device cuda:<GPU_ID>

## 2. SIKIM REMI+
python src/eval/prepare_test_captions_and_generate.py \
  --captions_jsonl datasets/captions/captions_gukak_s.jsonl \
  --tokenizer_vocab datasets/artifacts/vocab_remi_sikim.pkl \
  --model_ckpt model/fine_g_output/epoch_300/kot2m_model.bin \
  --device cuda:<GPU_ID>

# [기본 평가지표 계산]
## 1. base REMI
export REMI_VOCAB_PATH=datasets/artifacts/vocab_remi.pkl

python src/eval/evaluate.py \
--captions_file src/eval/test_generated_general.jsonl \
--tokenizer_vocab "$REMI_VOCAB_PATH" \
--device cuda:<GPU_ID> \
--use_cosiatec

## 2. SIKIM REMI+
export REMI_SIKIM_VOCAB_PATH=datasets/artifacts/vocab_remi_sikim.pkl

python src/eval/evaluate.py \
--captions_file src/eval/test_generated_sikim.jsonl \
--tokenizer_vocab "$REMI_SIKIM_VOCAB_PATH" \
--device cuda:<GPU_ID> \
--use_cosiatec

# [ESR평가지표 계산]
## 1. base REMI
python src/eval/ERR_eval.py \
--jsonl_path src/eval/test_generated_general.jsonl

## 2. SIKIM REMI+
python src/eval/ERR_eval.py \
--jsonl_path src/eval/test_generated_sikim.jsonl
