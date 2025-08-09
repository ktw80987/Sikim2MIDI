import torch
import torch.nn as nn

from transformers import AutoTokenizer
import pickle

from src._model_transformers import Transformer

from setproctitle import setproctitle
setproctitle('wjg980807_gen')

model_path = './model/fine_g_output/epoch_300/kot2m_model.bin'
tokenizer_path = './datasets/artifacts/vocab_remi_sikim.pkl'

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Using device: {device}')

with open(tokenizer_path, 'rb') as f:
    r_tokenizer = pickle.load(f)

vocab_size = len(r_tokenizer)
print('Vocab size: ', vocab_size)

model = Transformer(vocab_size, 768, 8, 512, 18, 1024, False, 8, device = device)
model.load_state_dict(torch.load(model_path, map_location = device))
model.eval()

tokenizer = AutoTokenizer.from_pretrained('KETI-AIR/ke-t5-base', use_fast = False)
print('Model loaded.')

src = '장구가 중심이 되는 이 창작국악 곡은 경쾌한 4/4 박자와 셋잇단음표 리듬으로 진행됩니다. 중고음역대에서 활기차고 명료한 반주를 제공합니다.'
print('Generating for prompt: ' + src)

inputs = tokenizer(src, return_tensors = 'pt', padding = True, truncation = True)
input_ids = nn.utils.rnn.pad_sequence(inputs.input_ids, batch_first = True, padding_value = 0).to(device)
attention_mask = nn.utils.rnn.pad_sequence(inputs.attention_mask, batch_first = True, padding_value = 0).to(device)

output = model.generate(input_ids, attention_mask, max_len = 500, temperature = 1.0)
output_list = output[0].tolist()

# print("Generated REMI tokens:\n", output_list)
# print("Token count:", len(output_list))

generated_midi = r_tokenizer.decode(output_list)
generated_midi.dump_midi('./output5.mid')

# CUDA_VISIBLE_DEVICES=0 python demo.py