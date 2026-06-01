# VLSP Machine Translation: Fine-Tuning & Error Analysis

An end-to-end Machine Translation pipeline built with PyTorch to fine-tune and evaluate the pre-trained `Helsinki-NLP/opus-mt-en-vi` model on the official VLSP dataset.

## 🚀 Core Contributions
- **Custom PyTorch Optimization Loop:** Implemented a low-level training loop handling `loss.backward()`, `optimizer.step()`, and linear learning rate warmup schedules instead of using high-level trainers.
- **Dynamic Data Pipeline:** Structured automated PyTorch DataLoaders with dynamic sequence padding, handling specific label padding tokens (`-100`) to exclude padding from loss calculations.
- **Decoding Parameter Tuning:** Optimised model generation quality during inference by tuning decoding configurations including Beam Search width (`num_beams=5`).
- **Granular Error Analysis:** Developed a programmatic post-evaluation framework leveraging **chrF** and **SacreBLEU** metrics to automatically classify translation flaws into structural anomalies (**Omission vs. Hallucination**) based on length ratios.

## 🛠️ Tech Stack
- **Frameworks:** PyTorch, Hugging Face (Transformers, Datasets)
- **Metrics & Visualization:** SacreBLEU, Evaluate, Matplotlib, Seaborn
- **Environment:** Google Colab (NVIDIA T4 GPU)
