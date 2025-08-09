import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from transformers import get_scheduler

from accelerate import Accelerator, DistributedDataParallelKwargs
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration

from tqdm import tqdm

import os, yaml, pickle, json, jsonlines, logging, time, math

from _model_transformers import Transformer
from _data_loader_remi import Text2MusicDataset
from torch.utils.data import random_split 

from setproctitle import setproctitle
setproctitle('wjg980807_g')

main_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

logger = get_logger(__name__)

# ----------------------------------------------------------------------------------------------------

#configs
config_file = f'{main_path}/src/configs/first_config.yaml'
with open(config_file, 'r') as f:
    configs = yaml.safe_load(f)

def point_replace(data, path):
    if isinstance(data, str):
        return data.replace('<point>', path)
    
    elif isinstance(data, dict):
        return {key: point_replace(value, path) for key, value in data.items()}
    
    elif isinstance(data, list):
        return [point_replace(item, path) for item in data]
    
    return data

configs = point_replace(configs, main_path)

model_config = configs['model']
training_config = configs['pre_training']

# First
batch_size = training_config['batch_size']
learning_rate = training_config['learning_rate']
epochs = training_config['epochs']

# Second
d_model = model_config['decoder_d_model']
nhead = model_config['decoder_num_heads']
num_layers = model_config['decoder_num_layers']
max_len = model_config['decoder_max_sequence_length']
use_moe = model_config['use_moe']
num_experts = model_config['num_experts']
dim_feedforward = model_config['decoder_intermediate_size']
gradient_accumulation_steps = training_config['gradient_accumulation_steps']
use_scheduler = training_config['use_scheduler']
checkpointing_steps = training_config['checkpointing_steps']
lr_scheduler_type = training_config['lr_scheduler_type']
num_warmup_steps = training_config['num_warmup_steps']
max_train_steps = training_config['max_train_steps']
with_tracking = training_config['with_tracking']
output_dir = training_config['output_dir']
per_device_train_batch_size = training_config['per_device_train_batch_size']
save_every = training_config['save_every']

tokenizer_path = os.path.join(configs['artifact_folder'], 'vocab_remi.pkl')
with open(tokenizer_path, 'rb') as f:
    tokenizer = pickle.load(f)
vocab_size = len(tokenizer)

caption_path = configs['raw_data']['commu']['caption_path']
with jsonlines.open(caption_path) as r:
    captions = list(r)

# ----------------------------------------------------------------------------------------------------

def collate_fn(batch):
    input_ids = [item[0].squeeze(0) for item in batch]
    input_ids = nn.utils.rnn.pad_sequence(input_ids, batch_first = True, padding_value = 0)

    attention_mask = [item[1].squeeze(0) for item in batch]
    attention_mask = nn.utils.rnn.pad_sequence(attention_mask, batch_first = True, padding_value = 0)

    labels = [item[2].squeeze(0) for item in batch]
    labels = nn.utils.rnn.pad_sequence(labels, batch_first = True, padding_value = 0)

    return input_ids, attention_mask, labels

project_config = ProjectConfiguration(project_dir = output_dir, logging_dir = output_dir if with_tracking else None)

accelerator = Accelerator(gradient_accumulation_steps = gradient_accumulation_steps,
                          mixed_precision = 'fp16',
                          kwargs_handlers = [DistributedDataParallelKwargs(find_unused_parameters = True)],
                          project_config = project_config)

logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger.info(accelerator.state, main_process_only = False)

if accelerator.is_main_process:
    os.makedirs(output_dir, exist_ok = True)
    accelerator.project_configuration.automatic_checkpoint_naming = False

accelerator.wait_for_everyone()

device = accelerator.device

with accelerator.main_process_first():
    dataset_path = configs['raw_data']['commu']['folder_path']
    
    full_dataset = Text2MusicDataset(
        configs = configs,
        captions = captions,
        remi_tokenizer = tokenizer,
        dataset_path = dataset_path,
        mode = 'train',
        shuffle = True
    )

    split_seed = 42
    train_ratio = 0.9
    train_len = int(len(full_dataset) * train_ratio)
    test_len = len(full_dataset) - train_len

    train_dataset, test_dataset = random_split(
        full_dataset,
        [train_len, test_len],
        generator = torch.Generator().manual_seed(split_seed)
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size = per_device_train_batch_size,
        shuffle = True,
        num_workers = 4,
        collate_fn = collate_fn,
        drop_last = True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size = per_device_train_batch_size,
        shuffle = False,
        num_workers = 4,
        collate_fn = collate_fn,
        drop_last = False,
    )

model = Transformer(vocab_size, d_model, nhead, max_len, num_layers, dim_feedforward, use_moe, num_experts, device = device)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

total_params = count_parameters(model)
print(f'Total number of trainable parameters: {total_params}')

optimizer = optim.Adam(model.parameters(), lr = learning_rate)
overrode_max_train_steps = False
num_update_steps_per_epoch = math.ceil(len(train_loader) / gradient_accumulation_steps)
print('num_update_steps_per_epoch', num_update_steps_per_epoch)
print('max_train_steps', max_train_steps)

if max_train_steps is None or max_train_steps == 'None':
    max_train_steps = epochs * num_update_steps_per_epoch
    print('max_train_steps', max_train_steps)
    overrode_max_train_steps = True
    num_warmup_steps = 20000

elif isinstance(max_train_steps, str):
    max_train_steps = int(max_train_steps)

lr_scheduler = get_scheduler(name = lr_scheduler_type, optimizer = optimizer, num_warmup_steps = num_warmup_steps, num_training_steps = max_train_steps)

model, optimizer, lr_scheduler = accelerator.prepare(model, optimizer, lr_scheduler)
dataloader = accelerator.prepare(train_loader)

if overrode_max_train_steps:
    max_train_steps = epochs * num_update_steps_per_epoch

epochs = math.ceil(max_train_steps / num_update_steps_per_epoch)

total_batch_size = per_device_train_batch_size * accelerator.num_processes * gradient_accumulation_steps
logger.info('***** Running training *****')
logger.info(f'  Num examples = {len(train_dataset)}')
logger.info(f'  Num Epochs = {epochs}')
logger.info(f'  Instantaneous batch size per device = {per_device_train_batch_size}')
logger.info(f'  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}')
logger.info(f'  Gradient Accumulation steps = {gradient_accumulation_steps}')
logger.info(f'  Total optimization steps = {max_train_steps}')

# ----------------------------------------------------------------------------------------------------

criterion = nn.CrossEntropyLoss(ignore_index = 0)

@torch.no_grad()
def evaluate_model(model, dataloader, device, criterion):
    model.eval()
    model.to(device)
    total_loss = 0
    total_tokens = 0

    for batch in dataloader:
        encoder_input, attention_mask, tgt = batch
        encoder_input = encoder_input.to(device)
        attention_mask = attention_mask.to(device)
        tgt = tgt.to(device)

        tgt_input = tgt[:, :-1]
        tgt_output = tgt[:, 1:]

        outputs = model(encoder_input, attention_mask, tgt_input)

        loss = criterion(outputs.view(-1, outputs.size(-1)), tgt_output.reshape(-1))
        total_loss += loss.item() * tgt_output.numel()
        total_tokens += tgt_output.numel()

    avg_loss = total_loss / total_tokens
    ppl = math.exp(avg_loss)
    return round(avg_loss, 4), round(ppl, 4)

def train_model_accelerate(model,
                           dataloader,
                           criterion,
                           num_epochs,
                           max_train_steps,
                           optimizer = None,
                           out_dir = None,
                           checkpointing_steps='epoch',
                           with_tracking = False,
                           save_every = 5,
                           device = 'cpu',
                           resume_from_epoch = None):

    if resume_from_epoch is not None:
        starting_epoch = resume_from_epoch
        completed_steps = starting_epoch * num_update_steps_per_epoch

        resume_path = os.path.join(out_dir, f'epoch_{resume_from_epoch}')
        accelerator.load_state(resume_path)

        logger.info(f'Resuming training from epoch {starting_epoch}, step {completed_steps}')
    else:
        starting_epoch = 0 # FIRST TRAIN
        completed_steps = 0
    
    progress_bar = tqdm(total = max_train_steps, initial = completed_steps, disable = not accelerator.is_local_main_process)

    model = model.to(device)
    model.train()

    best_loss = float('inf')

    if os.path.exists(f'{out_dir}/summary.jsonl'):
        with open(f'{out_dir}/summary.jsonl') as f:
            try:
                best_loss = min(json.loads(line)['train_loss'] for line in f)
                logger.info(f'Loaded previous best loss: {best_loss}')
            except Exception as e:
                logger.warning(f'Could not load previous best loss: {e}')

    for epoch in range(starting_epoch, num_epochs):

        total_loss = 0
        for step, batch in enumerate(dataloader):

            with accelerator.accumulate(model):
                
                encoder_input, attention_mask, tgt = batch
                encoder_input = encoder_input.to(device)
                attention_mask = attention_mask.to(device)
                
                tgt = tgt.to(device)
                tgt_input = tgt[:, :-1]
                tgt_output = tgt[:, 1:]

                if use_moe:
                    outputs, aux_loss = model(encoder_input, attention_mask, tgt_input)
                else:
                    outputs = model(encoder_input, attention_mask, tgt_input)
                    aux_loss = 0

                loss = criterion(outputs.view(-1, outputs.size(-1)), tgt_output.reshape(-1))
                loss += aux_loss
                total_loss += loss.detach().float()

                accelerator.backward(loss)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            if accelerator.sync_gradients:
                progress_bar.set_postfix({'Loss': loss.item()})
                progress_bar.update(1)
                completed_steps += 1

            if isinstance(checkpointing_steps, int):
                if completed_steps % checkpointing_steps == 0:
                    output_dir_step = f'step_{completed_steps}'

                    if out_dir is not None:
                        output_dir_step = os.path.join(out_dir, output_dir_step)

                    accelerator.save_state(output_dir_step)

            if completed_steps >= max_train_steps:
                break

        if accelerator.is_main_process:
            avg_loss = total_loss.item() / len(dataloader)
            try:
                ppl = round(math.exp(avg_loss), 4)
            except OverflowError:
                ppl = float('inf')

            result = {
                'epoch': epoch + 1,
                'step': completed_steps,
                'train_loss': round(avg_loss, 4),
                'train_ppl': ppl
            }

            with open(f'{out_dir}/summary.jsonl', 'a') as f:
                f.write(json.dumps(result) + '\n')

            logger.info(result)

            if checkpointing_steps == 'epoch' and ((epoch + 1) % save_every == 0 or (epoch + 1) == num_epochs):
                epoch_dir = os.path.join(out_dir, f'epoch_{epoch + 1}')
                os.makedirs(epoch_dir, exist_ok = True)

                accelerator.save_state(epoch_dir)

                unwrapped_model = accelerator.unwrap_model(model)
                epoch_bin_path = os.path.join(epoch_dir, f'kot2m_model.bin')
                torch.save(unwrapped_model.state_dict(), epoch_bin_path)
                logger.info(f'Model weights saved at: {epoch_bin_path}')

            if checkpointing_steps == 'best':
                if total_loss < best_loss:
                    best_loss = total_loss
                    best_dir = os.path.join(out_dir, 'best')
                    accelerator.save_state(best_dir)

                    unwrapped_model = accelerator.unwrap_model(model)
                    best_model_path = os.path.join(best_dir, f'kot2m_model.bin')
                    torch.save(unwrapped_model.state_dict(), best_model_path)
                    logger.info(f'Best model saved at: {best_model_path}')

    if accelerator.is_main_process:
        val_loss, val_ppl = evaluate_model(model, test_loader, device, criterion)
        logger.info(f'[Eval] Validation loss: {val_loss}, Validation PPL: {val_ppl}')
        with open(f'{out_dir}/summary.jsonl', 'a') as f:
            f.write(json.dumps({'eval_loss': val_loss, 'eval_ppl': val_ppl}) + '\n')

# START TRAIN
train_model_accelerate(model,
                       dataloader,
                       criterion,
                       num_epochs = epochs,
                       max_train_steps = max_train_steps,
                       optimizer = optimizer,
                       out_dir = output_dir,
                       checkpointing_steps = checkpointing_steps,
                       with_tracking = False,
                       save_every = save_every,
                       device = device)