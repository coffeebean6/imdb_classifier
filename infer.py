import os
import torch
from multiprocessing import cpu_count
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import get_peft_config, PeftModel, PeftConfig, get_peft_model, LoraConfig, TaskType

def tokenize_function(text):
    return tokenizer(
        text, 
        padding="max_length", 
        truncation=True, 
        max_length=512, 
        return_tensors='pt'
    )

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f' **** Device Type: {device}')
if torch.cuda.device_count() > 0:
    cuda_device_names = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
    print(f' **** Device Name: {cuda_device_names}')
os_cpu_cores = os.cpu_count()
cpu_cores = cpu_count()
print(f" **** CPU Cores: {os_cpu_cores}/{cpu_cores}")
print(f' **** Torch Version: {torch.__version__}')


class CFG:
    base_model = "distilbert-base-uncased"
    num_labels = 2
    state_dict_path = './models/model-2.0_e0-0.926.ckpt'    #'./models/model-2.0_e0-0.926.ckpt'


tokenizer = AutoTokenizer.from_pretrained(CFG.base_model)
base_model = AutoModelForSequenceClassification.from_pretrained(
    CFG.base_model, 
    num_labels=CFG.num_labels
)

lora_config = LoraConfig(
    r = 8,
    lora_alpha = 1, 
    target_modules=['q_lin', 'v_lin', 'lin1', 'lin2'],
    bias='all', #'lora_only',
    inference_mode=False,
    task_type=TaskType.SEQ_CLS,
    init_lora_weights="gaussian"
)

# Create LoRa Model
if CFG.state_dict_path != "":
    model = get_peft_model(base_model, lora_config)
    state_dict = torch.load(CFG.state_dict_path)
    model.load_state_dict(state_dict, strict=False)
    print(" **** Load model checkpoint:", CFG.state_dict_path)
else:
    model = base_model

# 单句推理
def infer_sentence(review_text):
    input_tensor = tokenize_function(review_text)
    input_tensor = input_tensor.to(device)

    with torch.no_grad():  
        output = model(**input_tensor)
        
    # 对output取softmax
    probabilities = torch.nn.functional.softmax(output.logits, dim=-1)
    print(f' **** Single Test: Softmaxed probabilities: {probabilities}')
    predicted_class = torch.argmax(probabilities, dim=-1).item()
    print(f' **** Single Test: Predicted class: {predicted_class}')
    return predicted_class
