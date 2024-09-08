import warnings
warnings.filterwarnings('ignore')
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import math
import time
import random
import numpy as np
import pandas as pd
from functools import partial
import argparse
from multiprocessing import cpu_count

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from datasets import Dataset
from transformers import AutoTokenizer, DataCollatorWithPadding, AutoModelForSequenceClassification
from peft import get_peft_config, PeftModel, PeftConfig, get_peft_model, LoraConfig, TaskType
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from metrics import AverageMeter, ProgressMeter
from schedulers import get_schedule_with_warmup_constant_decay, get_schedule_with_WCD_cosine_anneal


def set_seeds(seed):
    """Set seeds for reproducibility """
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True
    os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    
    
def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true'):
        return True
    elif v.lower() in ('no', 'false'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
    
    
def tokenize_function(sample, tokenizer):
    return tokenizer(
        sample["text"], 
        padding="max_length", 
        truncation=True, 
        max_length=512, 
        return_tensors='pt'
    )
    

def count_parameters(model):
    total_params = 0
    trainable_params = 0
    for p in model.parameters():
        total_params += p.numel()
        if p.requires_grad:
            trainable_params += p.numel()
    return total_params, trainable_params


def get_data_loaders():
    # Load data
    train_data = Dataset.from_pandas(pd.read_csv("./data/train-liam.csv"))
    valid_data = Dataset.from_pandas(pd.read_csv("./data/val-liam.csv"))
    test_data = Dataset.from_pandas(pd.read_csv("./data/test-liam.csv"))

    tokenizer = AutoTokenizer.from_pretrained(CFG.base_model)
    _tokenize_function = partial(tokenize_function, tokenizer=tokenizer)

    tokenized_train = train_data.map(_tokenize_function, batched=True)
    tokenized_valid = valid_data.map(_tokenize_function, batched=True)
    tokenized_test = test_data.map(_tokenize_function, batched=True)

    # 调整格式
    tokenized_train = tokenized_train.remove_columns(['text'])
    tokenized_valid = tokenized_valid.remove_columns(["text"])
    tokenized_test = tokenized_test.remove_columns(["text"])
    tokenized_train.set_format("torch", columns=["input_ids", "attention_mask", "label"])
    tokenized_valid.set_format("torch", columns=["input_ids", "attention_mask", "label"])
    tokenized_test.set_format("torch", columns=["input_ids", "attention_mask", "label"])

    # 定义一个数据填充器
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    train_loader = DataLoader(dataset=tokenized_train, batch_size=CFG.batch_size, num_workers=4, collate_fn=data_collator, pin_memory=True,
        shuffle=True, 
        drop_last=True
    )
    valid_loader = DataLoader(dataset=tokenized_valid, batch_size=CFG.batch_size, num_workers=4, collate_fn=data_collator, pin_memory=True,
        drop_last=False
    )
    test_loader = DataLoader(dataset=tokenized_test, batch_size=CFG.batch_size, num_workers=4, collate_fn=data_collator, pin_memory=True,
        drop_last=False
    )
    return train_loader, valid_loader, test_loader


def train(model, train_loader, optimizer, lr_scheduler, device, epoch):
    batch_time = AverageMeter('Time', ':6.3f') # 每个epoch会重新做平均
    metrics_loss = AverageMeter('Loss', ':.4e')
    metrics_acc = AverageMeter('Acc', ':6.2f')
    progress = ProgressMeter(len(train_loader), batch_time, metrics_loss, metrics_acc)
    
    model.train()
    end = time.time()
    for idx, batch in enumerate(train_loader):
        batch = {k: v.to(device) for k, v in batch.items()}
        target = batch['labels']
        
        outputs = model(**batch)
        loss = outputs.loss # 直接获取损失
        logits = outputs.logits
        correct_predictions = (torch.argmax(logits, dim=-1) == target).sum().item()
        
        loss.backward()
        optimizer.step()
        #lr_scheduler.step()
        optimizer.zero_grad()
        
        step = epoch * CFG.num_steps + idx
        step_loss = loss.item() / target.size(0)
        step_acc = correct_predictions / target.size(0)
        writer.add_scalar("LR", optimizer.param_groups[0]['lr'], step)
        writer.add_scalar("Loss", step_loss, step)
        writer.add_scalar("Acc", step_acc, step)
        batch_time.update(time.time() - end)
        metrics_loss.update(step_loss, target.size(0))
        metrics_acc.update(step_acc, target.size(0))
        end = time.time()
        
        if not idx % 100:
            progress.pr2int(idx)
            
# 模型评估
def eval(model, valid_loader, device, dataset_name='Valid'):
    model.eval()
    total_loss = 0.0
    correct_preds = 0
    total_samples = 0
    with torch.no_grad():
        for batch in tqdm(valid_loader, desc="Evaluating"):
            batch = {k: v.to(device) for k, v in batch.items()}
            target = batch['labels']
            
            outputs = model(**batch)
            loss = outputs.loss
            logits = outputs.logits
            
            correct_preds += (torch.argmax(logits, dim=-1) == target).sum().item()
            total_samples += target.size(0)
            total_loss += loss.item()
            
    avg_loss = total_loss / len(valid_loader)
    avg_acc = correct_preds / total_samples
    print(f" **** Eval {dataset_name} accurate: {avg_acc}")
    return avg_loss, avg_acc


# 模型测试
def test(base_model, lora_config, ckpt_name, device, test_loader, dataset_name):
    model = get_peft_model(base_model, lora_config)
    state_dict = torch.load(ckpt_name)
    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    return eval(model, test_loader, device, dataset_name)
    

class CFG:
    tag = 2.00
    base_model = "distilbert-base-uncased"
    num_epochs=4
    batch_size=12
    num_steps = math.ceil(35000 / batch_size)
    num_training_steps = num_epochs * num_steps
    ## 继续训练参数
    num_epochs_start=0
    last_epoch=-1
    last_checkpoint='' #'./models/model-2.0_e1-0.94'


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f' **** Device Type: {device}')
if torch.cuda.device_count() > 0:
    cuda_device_names = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
    print(f' **** Device Name: {cuda_device_names}')
os_cpu_cores = os.cpu_count()
cpu_cores = cpu_count()
print(f" **** CPU Cores: {os_cpu_cores}/{cpu_cores}")
print(f' **** Torch Version: {torch.__version__}')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='LoRA parameters configuration')
    parser.add_argument('--lora_r', type=int, default=8, help='Rank for LoRA layers')
    parser.add_argument('--lora_alpha', type=int, default=1, help='Alpha for LoRA layers')
    parser.add_argument('--lora_query', type=str2bool, default=True, help='Apply LoRA to query')
    parser.add_argument('--lora_key', type=str2bool, default=False, help='Apply LoRA to key')
    parser.add_argument('--lora_value', type=str2bool, default=True, help='Apply LoRA to value')
    parser.add_argument('--lora_projection', type=str2bool, default=False, help='Apply LoRA to projection')
    parser.add_argument('--lora_mlp', type=str2bool, default=True, help='Apply LoRA to MLP')
    parser.add_argument('--lora_head', type=str2bool, default=False, help='Apply LoRA to head')
    args = parser.parse_args()
    
    module_list = ['q_lin', 'v_lin']
    if args.lora_key:
        module_list.append('k_lin')
    if args.lora_projection:
        module_list.append('out_lin')
    if args.lora_mlp:
        module_list.append('lin1')
        module_list.append('lin2')
    if args.lora_head:
        module_list.append('pre_classifier')
        module_list.append('classifier')
    print(f" **** Target Modules: {module_list}")

    set_seeds(seed=2024)
    
    # Load data
    train_loader, valid_loader, test_loader = get_data_loaders()

    base_model = AutoModelForSequenceClassification.from_pretrained(
        CFG.base_model, 
        num_labels=2
    )
    
    for param in base_model.parameters():
        param.requires_grad = False
        
    lora_config = LoraConfig(
        r = args.lora_r,
        lora_alpha = args.lora_alpha, 
        target_modules=module_list,
        bias='all', #'lora_only',
        inference_mode=False,
        task_type=TaskType.SEQ_CLS,
        init_lora_weights="gaussian"
    )

    # Create LoRa Model
    model = get_peft_model(base_model, lora_config)

    if CFG.last_checkpoint != "":
        state_dict = torch.load(CFG.last_checkpoint + ".ckpt")
        model.load_state_dict(state_dict, strict=False)
        print(" **** Load model checkpoint:", CFG.last_checkpoint)

    # Trainable Parameters
    print(model)
    total_params, trainable_params = count_parameters(model)
    print(" **** Total number of parameters: ", total_params)
    print(f" **** Number of trainable parameters: {trainable_params} ({trainable_params/total_params*100:.2f}%)")
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=9e-5, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-2)
    if CFG.last_checkpoint != "":
        optimizer.load_state_dict(torch.load(CFG.last_checkpoint + "-optimizer.ckpt"))
        print(" **** Load optimizer checkpoint:", CFG.last_checkpoint)

    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CFG.num_epochs, eta_min=1e-6)

    writer = SummaryWriter(f'./logs/{CFG.tag}')
    
    best_acc = 0.0
    best_ckpt_name = CFG.last_checkpoint
    for epoch in range(CFG.num_epochs_start, CFG.num_epochs):
        print(f"Epoch {epoch + 1}/{CFG.num_epochs}")
        
        train(model, train_loader, optimizer, lr_scheduler, device, epoch)
        val_loss, val_acc = eval(model, valid_loader, device)
        lr_scheduler.step()
        
        if val_acc > best_acc:
            best_acc = val_acc
            best_ckpt_name = f"./models/model-{CFG.tag}_e{epoch}-{best_acc}"
            print("**** Save checkpoint: ", best_ckpt_name)
            torch.save(
                dict([(k, v.cpu()) for k, v in model.named_parameters() if v.requires_grad]),
                best_ckpt_name + ".ckpt"
            )
            torch.save(optimizer.state_dict(), best_ckpt_name + "-optimizer.ckpt")
        else:
            break    # early stop
            
    writer.close()

    print(f" **** best ckpt: {best_ckpt_name}")
    _, test_acc = test(base_model, lora_config, best_ckpt_name + ".ckpt", device, test_loader, "Test")

    # Print settings and results
    print("------------------------------------------------")
    for arg in vars(args):
        print(f'{arg}: {getattr(args, arg)}')
    print()
    try:
        print(f"Val acc:   {val_acc*100:2.2f}%")
    except NameError:
        pass
    print(f"Test acc:  {test_acc*100:2.2f}%")
    print("------------------------------------------------")

