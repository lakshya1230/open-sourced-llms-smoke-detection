import habana_frameworks.torch as ht
import habana_frameworks.torch.core as htcore

from optimum.habana.transformers.modeling_utils import adapt_transformers_to_gaudi
adapt_transformers_to_gaudi()

from transformers import LlavaNextProcessor, AutoProcessor, AutoConfig, Qwen2VLForConditionalGeneration
import torch
from qwen_vl_utils import process_vision_info
from PIL import Image
import requests
import os
import time
import re


device = torch.device("hpu")
args_model_name_or_path = "workspace/models/Qwen2-VL-7B-Instruct"
model_type = AutoConfig.from_pretrained(args_model_name_or_path).model_type

min_pixels = 256 * 28 * 28
max_pixels = 1280 * 28 * 28

print("Loading the processor")
processor = AutoProcessor.from_pretrained(args_model_name_or_path, min_pixels=min_pixels, max_pixels=max_pixels)

print("Loading the model")
model = Qwen2VLForConditionalGeneration.from_pretrained(
    args_model_name_or_path,
    torch_dtype=torch.bfloat16
)

print("Transfering to hpu")
model.to("hpu")

fire_sequences_root = "/workspace/Full_data/test_sequences"
# Initialize results dictionary
prediction_results = {}

# Get all fire sequence folders
fire_sequence_folders = sorted([
    os.path.join(fire_sequences_root, folder)
    for folder in os.listdir(fire_sequences_root)
    if os.path.isdir(os.path.join(fire_sequences_root, folder))
])

print(f"Found {len(fire_sequence_folders)} fire sequence folders.")

# Start processing each fire sequence folder
start = time.perf_counter()

batch_size = 1
generate_kwargs = {"max_new_tokens": 200, "do_sample": False}
prediction_results_per_image_file = {}

# Initialize confusion matrix metrics
metrics = {
    "true_positive": 0,
    "false_negative": 0,
    "false_positive": 0,
    "true_negative": 0
}

# Setting up the conversation for the wildfire detection
fire_conversation = [
    {
        "role": "user",
        "content": [
           {"type": "text", "text": "You are an expert in detecting wildfire smokes starting from their ignition. Can you see one in the image? Respond in simple 'YES' or 'NO'. No explanation needed."},
           {"type": "image"},
        ],
    }
]

fire_prompt = processor.apply_chat_template(fire_conversation, tokenize=False, add_generation_prompt=True)

for fire_sequence_folder in fire_sequence_folders:
    fire_sequence_name = os.path.basename(fire_sequence_folder)

    prediction_results[fire_sequence_name] = {
        "original": {"before": 0, "after": 0},
        "prediction": {"before": 0, "after": 0}
    }

    fire_image_paths = sorted([
        os.path.join(fire_sequence_folder, file)
        for file in os.listdir(fire_sequence_folder)
        if file.endswith(".jpg")
    ])

    print(f"Processing {fire_sequence_name} with {len(fire_image_paths)} images...")

    # Batch processing
    for idx in range(0, len(fire_image_paths), batch_size):
        print("In the here")
        batch_images = [Image.open(image_path) for image_path in fire_image_paths[idx : idx + batch_size]]
        batch_paths = fire_image_paths[idx : idx + batch_size]
        print("creating messages")
        messages = [[{
        "role": "user",
        "content": [
            {"type": "image", "image": current_im},
            {"type": "text", "text": "You are an expert in detecting wildfire smokes starting from their ignition. Can you see one in the image? Respond in simple 'YES' or 'NO'. No explanation needed."},
        ],
        }] for current_im in batch_paths]

        print("applying chat templates")
        texts = [
        processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
        for msg in messages
        ]

        print("Processing...")
        image_inputs, video_inputs = process_vision_info(messages)

        print("starting processing")

        current_batch_inputs = processor(
            images=image_inputs, 
            text=texts, 
            videos=video_inputs,
            padding=True, 
            return_tensors="pt"
        ).to(model.device)

        print("Sending input to hpu")
        current_batch_inputs = current_batch_inputs.to("hpu")
        print("starting generation")
        current_batch_generate_ids = model.generate(**current_batch_inputs, max_new_tokens=150)

        print("doing decoding")
        current_batch_results = processor.batch_decode(
            current_batch_generate_ids, 
            skip_special_tokens=True, 
            clean_up_tokenization_spaces=False
        )

        print("Current result: ", current_batch_results)

        for image_path, result in zip(batch_paths, current_batch_results):
            current_fire_image_path = os.path.basename(image_path)
            processed_result = result.upper().split("ASSISTANT")[-1].strip().upper()
            print("RES: ", result)
            print("PROCESSED: ", processed_result)
            print(f"File: {current_fire_image_path}, Result: {processed_result}")

            is_fire = "+" in current_fire_image_path
            prediction_is_fire = "YES" in processed_result
     
            if is_fire and prediction_is_fire:
                metrics["true_positive"] += 1
            elif is_fire and not prediction_is_fire:
                metrics["false_negative"] += 1
            elif not is_fire and prediction_is_fire:
                metrics["false_positive"] += 1
            elif not is_fire and not prediction_is_fire:
                metrics["true_negative"] += 1

            if is_fire:
                prediction_results[fire_sequence_name]["original"]["after"] += 1
            else:
                prediction_results[fire_sequence_name]["original"]["before"] += 1

            if prediction_is_fire:
                prediction_results[fire_sequence_name]["prediction"]["after"] += 1
            else:
                prediction_results[fire_sequence_name]["prediction"]["before"] += 1

            prediction_results_per_image_file[image_path] = processed_result

end = time.perf_counter()
duration = end - start
print(f"Total duration: {duration:.2f} seconds")

# Print summary of results
for seq_name, results in prediction_results.items():
    print(f"\nSummary for {seq_name}:")
    print(f"  Original - Before: {results['original']['before']}, After: {results['original']['after']}")
    print(f"  Prediction - Before: {results['prediction']['before']}, After: {results['prediction']['after']}")

# Print confusion matrix metrics
print("\nConfusion Matrix Metrics:")
print(f"  True Positives: {metrics['true_positive']}")
print(f"  False Negatives: {metrics['false_negative']}")
print(f"  False Positives: {metrics['false_positive']}")
print(f"  True Negatives: {metrics['true_negative']}")