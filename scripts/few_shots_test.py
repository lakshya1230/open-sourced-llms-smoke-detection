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
import random


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




# Go to the train sequences
fire_sequences_root = "/workspace/Full_data/test_sequences/train_sequences/train_fires_final"
# Initialize results dictionary
prediction_results = {}

# Get all fire sequence folders
fire_sequence_folders = sorted([
    os.path.join(fire_sequences_root, folder)
    for folder in os.listdir(fire_sequences_root)
    if os.path.isdir(os.path.join(fire_sequences_root, folder))
])

print(f"Found {len(fire_sequence_folders)} fire sequence folders.")

first_5_sequences = fire_sequence_folders[:5]

# Randomly select 5 sequences for few shots training.
random.seed(42)
few_shot_sequences = random.sample(fire_sequence_folders, 1)

few_shot_images = []
labels = ["NO", "YES", "YES"]

print("Collecting Few Shot Images...")

for seq_folder in few_shot_sequences:
    # Get all images in the sequence
    seq_images = sorted([
        f for f in os.listdir(seq_folder) 
        if f.endswith(".jpg")
    ])
    
    # Select first, +0000.jpg, and last images
    key_images = [
        seq_images[0],  # First image (NO)
        next(f for f in seq_images if f.endswith("+00000.jpg")),  # Transition
        seq_images[-1]   # Last image (YES)
    ]
    
    # Add to few-shot set with labels
    for img_path, label in zip(key_images, labels):
        few_shot_images.append({
            "path": os.path.join(seq_folder, img_path),
            "label": label
        })


few_shot_prompt = ""

few_shot_prompt += f"""
[SYSTEM]
You are a wildfire detection expert
Analyze images and respond with "YES" if smoke is detected, or "NO" if not. Only output these exact words.
Use these examples below : 
[/SYSTEM]
"""

for example in few_shot_images:
    img = Image.open(example["path"])
    few_shot_prompt += f"""
[USER]
<image>Image: {os.path.basename(example['path'])}
Question: Is there any smoke plume in the above image? Answer strictly 'YES' or 'NO'.
[/USER]
[ASSISTANT] 
{example['label']}<|endoftext|>
"""

# Add instruction for new images
few_shot_prompt += """
[USER]
Now, answer the following question.
<image>
Question: Is there any smoke plume in the above image? Answer strictly 'YES' or 'NO'.
[/USER]
[ASSISTANT]
"""

# print(few_shot_prompt)

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

for fire_sequence_folder in first_5_sequences:
    if fire_sequence_folder in few_shot_sequences:
        continue 
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
    for img_path in fire_image_paths:
        # Combine few-shot examples with new image

        full_prompt = few_shot_prompt.replace("<image>", f"<image>{Image.open(img_path)}")
        
        # Run inference
        inputs = processor(
            text=full_prompt,
            images=Image.open(img_path),
            return_tensors="pt"
        ).to("hpu")
        
        outputs = model.generate(**inputs, max_new_tokens=1, eos_token_id=processor.tokenizer.eos_token_id,  # Force stop
    do_sample=False,
    num_beams=1)

        raw_output = processor.tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(raw_output)
        clean_answer = raw_output.split()[0].strip().upper()  # Takes "NO" from "NO /NO..."
        final_answer = "YES" if clean_answer == "YES" else "NO"

        is_fire = "+" in img_path
        prediction_is_fire = (final_answer == "YES")

        print(img_path)
        print(clean_answer)
        print(prediction_is_fire)
        print("================")

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
print(f" True Positives: {metrics['true_positive']}")
print(f" False Negatives: {metrics['false_negative']}")
print(f" False Positives: {metrics['false_positive']}")
print(f" True Negatives: {metrics['true_negative']}")