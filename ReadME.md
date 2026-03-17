# 🎼 Sikim2MIDI

음악 캡션을 기반으로 MIDI를 생성하는 모델 학습 및 평가 파이프라인입니다.

------------------------------------------------------------------------

## 🚀 사전학습 (Pre-training)

``` bash
accelerate launch --multi_gpu --num_processes=4 pre_train.py
```

------------------------------------------------------------------------

## 🎯 파인튜닝 (Fine-tuning)

``` bash
accelerate launch --multi_gpu --num_processes=4 fine_train.py
```

------------------------------------------------------------------------

## 🧪 평가 데이터 생성

출력 형식:\
\[id, test_data_캡션, 생성된 MIDI, test_data_midi\]

### 1. Base REMI

``` bash
python src/eval/prepare_test_captions_and_generate.py \
  --captions_jsonl datasets/captions/captions_gukak_f.jsonl \
  --tokenizer_vocab datasets/artifacts/vocab_remi.pkl \
  --model_ckpt model/fine_output/epoch_300/kot2m_model.bin \
  --device cuda:<GPU_ID>
```

### 2. SIKIM REMI+

``` bash
python src/eval/prepare_test_captions_and_generate.py \
  --captions_jsonl datasets/captions/captions_gukak_s.jsonl \
  --tokenizer_vocab datasets/artifacts/vocab_remi_sikim.pkl \
  --model_ckpt model/fine_g_output/epoch_300/kot2m_model.bin \
  --device cuda:<GPU_ID>
```

------------------------------------------------------------------------

## 📊 기본 평가지표 계산

### 1. Base REMI

``` bash
export REMI_VOCAB_PATH=datasets/artifacts/vocab_remi.pkl

python src/eval/evaluate.py \
  --captions_file src/eval/test_generated_general.jsonl \
  --tokenizer_vocab "$REMI_VOCAB_PATH" \
  --device cuda:<GPU_ID> \
  --use_cosiatec
```

### 2. SIKIM REMI+

``` bash
export REMI_SIKIM_VOCAB_PATH=datasets/artifacts/vocab_remi_sikim.pkl

python src/eval/evaluate.py \
  --captions_file src/eval/test_generated_sikim.jsonl \
  --tokenizer_vocab "$REMI_SIKIM_VOCAB_PATH" \
  --device cuda:<GPU_ID> \
  --use_cosiatec
```

------------------------------------------------------------------------

## 📈 ESR 평가지표 계산

### 1. Base REMI

``` bash
python src/eval/ERR_eval.py \
  --jsonl_path src/eval/test_generated_general.jsonl
```

### 2. SIKIM REMI+

``` bash
python src/eval/ERR_eval.py \
  --jsonl_path src/eval/test_generated_sikim.jsonl
```

------------------------------------------------------------------------

## 📝 Notes

-   `<GPU_ID>`는 사용 가능한 GPU 번호로 변경하세요.
-   accelerate 설정이 사전에 완료되어 있어야 합니다.
-   vocab 및 모델 경로는 환경에 맞게 수정하세요.
