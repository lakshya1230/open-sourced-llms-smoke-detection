import torch
import habana_frameworks.torch.core as htcore
import habana_frameworks.torch as ht
from transformers import AutoProcessor, AutoConfig, AutoTokenizer, Qwen2VLForConditionalGeneration, BitsAndBytesConfig
from optimum.habana.transformers.modeling_utils import adapt_transformers_to_gaudi
from qwen_vl_utils import process_vision_info
adapt_transformers_to_gaudi()

def print_memory_footprint():
    if torch.hpu.is_available():
        print(f"Memory Allocated in GB: {ht.hpu.memory_allocated() / 1024**3:.2f}")
        print(f"Memory Reserved in GB: {ht.hpu.memory_reserved() / 1024**3:.2f}")
        print(f"Memory Maximum in GB: {ht.hpu.max_memory_allocated() / 1024**3:.2f}")
    else:
        print("HPU not detected!")

device = torch.device("hpu")

args_model_name_or_path = "workspace/models/Qwen2-VL-7B-Instruct"
model_type = AutoConfig.from_pretrained(args_model_name_or_path).model_type
min_pixels = 256 * 28 * 28
max_pixels = 1280 * 28 * 28

bnb_config = BitsAndBytesConfig(
    load_in_8bit=True,
    bnb_8bit_compute_dtype=torch.bfloat16,  # Compute in FP16
)


print("Loading the processor")
processor = AutoProcessor.from_pretrained(args_model_name_or_path, min_pixels=min_pixels, max_pixels=max_pixels)

print("Loading the model")
model = Qwen2VLForConditionalGeneration.from_pretrained(
    args_model_name_or_path,
    device_map="hpu",
    torch_dtype=torch.bfloat16,
    quantization_config=bnb_config
)


print("Memory Specs after loading the model: ")
print_memory_footprint()

img_path1 = "workspace/Full_data/test_sequences/train_sequences/train_fires_final/20160604_FIRE_rm-n-mobo-c/1465063200_-02400.jpg"
img_path2 = "workspace/Full_data/test_sequences/train_sequences/train_fires_final/20160604_FIRE_rm-n-mobo-c/1465063260_-02340.jpg"


messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "image": img_path1},
            {"type": "text", "text": "Is there wildfire smoke in this image? Answer strictly in 'YES' or 'NO'?"}
        ]
    }, 
    {
        "role": "user",
        "content": [
            {"type": "image", "image": img_path2},
            {"type": "text", "text": "Is there wildfire smoke in this image? Answer strictly in 'YES' or 'NO'?"}
        ]
    }   
]



with torch.no_grad():
    ht.hpu.enable_recompute_sdp(True)

    text_1 = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs_1, video_inputs_1 = process_vision_info(messages)
    input_1 = processor(
            text=[text_1],
            images=image_inputs_1,
            videos = video_inputs_1,
            return_tensors="pt",
            padding= True
        ).to(model.device)
    output_1 = model.generate(**input_1, max_new_tokens=5)
    result_1 = processor.tokenizer.decode(output_1[0], skip_special_tokens=True)

    print(model.config.use_cache)
    print_memory_footprint()

