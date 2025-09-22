import torch
import habana_frameworks.torch.core as htcore
import habana_frameworks.torch as ht
from transformers import AutoProcessor, AutoConfig, LlavaForConditionalGeneration
from optimum.habana.transformers.modeling_utils import adapt_transformers_to_gaudi
adapt_transformers_to_gaudi()


def print_memory_footprint():
    if torch.hpu.is_available():
        print(f"Memory Allocated in GB: {ht.hpu.memory_allocated() / 1024**3:.2f}")
        print(f"Memory Reserved in GB: {ht.hpu.memory_reserved() / 1024**3:.2f}")
        print(f"Memory Maximum in GB: {ht.hpu.max_memory_allocated() / 1024**3:.2f}")
    else:
        print("HPU not detected!")

print("Memory Specs before  loading the model: ")
print_memory_footprint()

device = torch.device("hpu")
args_model_name_or_path = "workspace/models/llava-1.5-7b-hf"
print("Loading the processor")
processor = AutoProcessor.from_pretrained(args_model_name_or_path)

print("Loading the model")
model = LlavaForConditionalGeneration.from_pretrained(
    args_model_name_or_path,
    torch_dtype=torch.bfloat16
)
print("Transfering to hpu")
model.to("hpu")

print("Memory Specs after loading the model: ")
print_memory_footprint()


input_1 = processor(
    text=["Hi My name is Lakshya,"],
    return_tensors="pt",
    padding= True
).to(model.device)



input_2 = processor(
    text=["What is my name ?"],
    return_tensors="pt",
    padding= True
).to(model.device)

print(model)

def caculate_kv_cache_size(model):
    # bfloat is 2 bytes
    dtype_size = 2

    kv_cache_elements = sum(k.numel() + v.numel() for layer in model.language_model.layers if hasattr(layer.self_attn, "past_key_value"))
    kv_cache_memory = kv_cache_elements * dtype_size
    print(f"Total KV Cache Memory: {kv_cache_memory / (1024 ** 2):.2f} MB")

with torch.no_grad():

    output_1 = model.generate(**input_1, max_new_tokens=20)
    result_1 = processor.tokenizer.decode(output_1[0], skip_special_tokens=True, max_new_tokens=20)

    print("Result ", result_1)
    caculate_kv_cache_size(model)

    output_2 = model.generate(**input_2, max_new_tokens=20)
    result_2 = processor.tokenizer.decode(output_2[0], skip_special_tokens=True, max_new_tokens=20)

    print("Result ", result_2)
    caculate_kv_cache_size(model)






