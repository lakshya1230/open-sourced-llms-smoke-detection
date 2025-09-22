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

# Randomly select 3 sequences for few shots training.
random.seed(42)
few_shot_sequences = random.sample(fire_sequence_folders, 3)

few_shot_examples = []
labels = ["NO", "YES", "YES"]

print("Collecting Few Shot Images...")

for seq_folder in few_shot_sequences:
    # Get all images in the sequence
    seq_images = sorted([
        f for f in os.listdir(seq_folder) 
        if f.endswith(".jpg")
    ])
    
    # Select Image[0], Image[+0000], and Image[-1]
    selected_images = [
        seq_images[0],  
        next(f for f in seq_images if f.endswith("+00000.jpg")), 
        seq_images[-1]  
    ]
    
    # Add to few-shot set with labels
    for img_path, label in zip(selected_images, labels):
        few_shot_examples.append({
            "image": os.path.join(seq_folder, img_path),
            "question": "Is there wildfire smoke in this image? Answer strictly in 'YES' or 'NO' ? ",
            "answer": label
        })


few_shot_messages = []

for few_shot_example in few_shot_examples:
    few_shot_messages.append({
        "role": "user",
        "content": [
            {"type": "image", "image": few_shot_example["image"]},
            {"type": "text", "text": few_shot_example["question"]},
        ],
    })
    few_shot_messages.append({
        "role": "assistant",
        "content": few_shot_example["answer"]
    })


def reload_context(few_shot_messages):

    print("Reloading Context")
    print(f"Current memory: {ht.hpu.memory_allocated() / 1e9:.2f} GB")
    print(f"Peak memory: {ht.hpu.max_memory_allocated() / 1e9:.2f} GB")

    for i in range(0,3):

        text = processor.apply_chat_template(few_shot_messages[3*i : 3*i + 3], tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(few_shot_messages[3*i : 3*i + 3])

        inputs = processor(
            text=[text],
            images=image_inputs,
            videos = video_inputs,
            return_tensors="pt",
            padding= True
        ).to(model.device)

        outputs = model.generate(**inputs, max_new_tokens=10)
        result = processor.tokenizer.decode(outputs[0], skip_special_tokens=True)

    print("Result : ", result)

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

for fire_sequence_folder in fire_sequence_folders:

    reload_context(few_shot_messages)

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

        message = [{
            "role": "user",
            "content": [
                {"type": "image", "image": img_path},
                {"type": "text", "text": "Is there wildfire smoke in this image? Answer strictly in 'YES' or 'NO' ? "},
                {"type": "text", "text": "If YES, describe the smoke location ? "}
            ]
        }]

        print(f"Current memory: {ht.hpu.memory_allocated() / 1e9:.2f} GB")
        print(f"Peak memory: {ht.hpu.max_memory_allocated() / 1e9:.2f} GB")

        text = processor.apply_chat_template(message, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(message)


        # Run inference
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos = video_inputs,
            return_tensors="pt",
            padding= True
        ).to(model.device)


        print("Inputs", inputs)
        
        outputs = model.generate(**inputs, max_new_tokens=20)

        result = processor.tokenizer.decode(outputs[0], skip_special_tokens=True)

        print("Result : ", result)

        processed_result = result.upper().split("ASSISTANT")[-1].strip().split(",")[0].upper()
        final_answer = "YES" if "YES" in processed_result else "NO"

        is_fire = "+" in img_path
        prediction_is_fire = (final_answer == "YES")

        print("Image Path: ", img_path)
        print("Final Answer: ", final_answer)
        print("True Answer : ", is_fire)
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

metrics['precision'] = metrics['true_positive'] / (metrics['true_positive'] + metrics['false_positive'])
metrics['recall'] = metrics['true_positive'] / (metrics['true_positive'] + metrics['false_negative'])
metrics['f1_score'] = 2 * metrics['precision'] * metrics['recall'] / ( metrics['precision'] + metrics['recall'] )

# Print confusion matrix metrics
print("\nConfusion Matrix Metrics:")
print(f" True Positives: {metrics['true_positive']}")
print(f" False Negatives: {metrics['false_negative']}")
print(f" False Positives: {metrics['false_positive']}")
print(f" True Negatives: {metrics['true_negative']}")
print(f" Precision: {metrics['precision']}")
print(f" Recall: {metrics['recall']}")
print(f" F1-Score: {metrics['f1_score']}")
