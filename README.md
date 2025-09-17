# open-sourced-llms-smoke-detection
This project represents a cutting-edge application of Artificial Intelligence for environmental protection. It focuses on the early detection of wildfire smoke plumes by analyzing sequential satellite or camera imagery. The core innovation lies in leveraging advanced Vision-Language Models (VLMs) like Qwen2-VL, fine-tuned and prompted to act as expert analysts for identifying the earliest signs of wildfire ignition.

## Architecture


<img width="3071" height="3840" alt="image" src="https://github.com/user-attachments/assets/a6b6374e-59fd-4316-a6a5-9bda1665bf7b" />




**This project implements a sophisticated AI pipeline that comprises**:

Processes Temporal Data: It takes sequences of images (e.g., 60 frames) that capture the evolution of a landscape over time, from a pre-ignition state to the point where smoke is clearly visible.

Leverages State-of-the-Art Models: It utilizes powerful pre-trained VLMs from Hugging Face (specifically Qwen2-VL-7B-Instruct). These models are uniquely capable of understanding and reasoning about the complex visual content within images in the context of a textual prompt.

Optimizes for Performance: The entire system is optimized to run on HPU/GPU ensuring efficient processing and making deployment on a larger scale more feasible.

We employ various advanced AI techniques to explore the VLM's power:

- Zero-Shot Learning: Asking the model to detect smoke directly without any examples, testing its innate capability.

- Few-Shot Learning: Providing the model with a few labeled examples within the prompt to guide its reasoning and improve accuracy.

- Fine-Tuning: Further training the pre-trained model on a dataset of labeled smoke sequences to specialize it for this specific task, potentially yielding the highest performance.

## Data Preparation:

Videos are processed into sequences of 60 frames(images).

Each sequence is labeled as Positive (contains smoke) or Negative (no smoke).

These labeled sequences are stored for training and unlabeled sequences for evaluation.

## Evaluation

<img width="3840" height="2642" alt="image" src="https://github.com/user-attachments/assets/3d142327-d07b-4a69-b0a4-fb35aa2ae831" />


