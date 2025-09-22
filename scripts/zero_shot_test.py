import habana_frameworks.torch as ht
import habana_frameworks.torch.core as htcore

from optimum.habana.transformers.modeling_utils import adapt_transformers_to_gaudi
adapt_transformers_to_gaudi()

from transformers import AutoProcessor, AutoConfig, LlavaForConditionalGeneration
import torch
from PIL import Image
import requests
import os
import time
import re
import random


device = torch.device("hpu")
args_model_name_or_path = "workspace/models/llava-1.5-7b-hf"
model_type = AutoConfig.from_pretrained(args_model_name_or_path).model_type

min_pixels = 256 * 28 * 28
max_pixels = 1280 * 28 * 28

print("Loading the processor")
processor = AutoProcessor.from_pretrained(args_model_name_or_path, min_pixels=min_pixels, max_pixels=max_pixels)

print("Loading the model")
model = LlavaForConditionalGeneration.from_pretrained(
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
 
time_to_detect = []

for fire_sequence_folder in fire_sequence_folders :

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

    is_first_tp = False

    # Batch processing
    for img_path in fire_image_paths:

        image_curr = Image.open(img_path)
        message = [{
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": "Do you see any smoke plume in this image? Answer in YES or NO and describe the location"},
            ]
        }]

        print(f"Current memory: {ht.hpu.memory_allocated() / 1e9:.2f} GB")
        print(f"Peak memory: {ht.hpu.max_memory_allocated() / 1e9:.2f} GB")

        prompt = processor.apply_chat_template(message, tokenize=False, add_generation_prompt=True)

        inputs = processor(
            text=prompt,
            images=image_curr,
            return_tensors="pt",
            padding= True
        ).to(model.device)
        
        outputs = model.generate(**inputs, max_new_tokens=50,  use_cache=True)
        result = processor.tokenizer.decode(outputs[0], skip_special_tokens=True)

        processed_result = result.upper().split("ASSISTANT")[-1].strip().split(",")[0].upper()

        final_answer = "YES" if "YES" in processed_result else "NO"

        is_fire = "+" in img_path
        prediction_is_fire = (final_answer == "YES")

        print("Image Path: ", img_path)
        print("Predicted Answer : ", prediction_is_fire)
        print("Correct Answer : ", is_fire)
        print("================")
        print("Time to Detect :", time_to_detect)

        if is_fire and prediction_is_fire:
            if is_first_tp == False:
                is_first_tp = True
                time_to_detect.append(int(img_path.split('_')[-1].split('.')[0]) / 60)
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

metrics['precision'] = metrics['true_positive'] / (metrics['true_positive'] + metrics['false_positive'] + 0.0000001)
metrics['recall'] = metrics['true_positive'] / (metrics['true_positive'] + metrics['false_negative'] + 0.0000001)
metrics['f1_score'] = 2 * metrics['precision'] * metrics['recall'] / ( metrics['precision'] + metrics['recall'] )
metrics['time_to_detect'] = sum(time_to_detect) / len(time_to_detect)

# Print confusion matrix metrics
print("\nConfusion Matrix Metrics:")
print(f" True Positives: {metrics['true_positive']}")
print(f" False Negatives: {metrics['false_negative']}")
print(f" False Positives: {metrics['false_positive']}")
print(f" True Negatives: {metrics['true_negative']}")
print(f" Precision: {metrics['precision']}")
print(f" Recall: {metrics['recall']}")
print(f" F1-Score: {metrics['f1_score']}")
print(f" Time to Detect: {metrics['time_to_detect']}")
