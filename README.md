# Ghibli IA Generator Model 🎨

Fine-tuning a Stable Diffusion anime model with LoRA to replicate the watercolor, nostalgic style of Studio Ghibli on custom character generations.

---

## 1. Project Overview

This project is a case study for the course **Deep Learning & Intelligent Systems**, where we fine-tuned a LoRA adapter on top of the anime model `waifu-diffusion/wd-1-5-beta2` to transfer the **Studio Ghibli** style to new concepts (e.g., anime characters that are *not* from Ghibli).

Our main research question was:

> *Is careful data curation more important than hyperparameter tuning when teaching a diffusion model a new artistic style?*

Our short answer: **yes — balanced, curated data had a bigger impact than aggressive hyperparameter tweaks.**

---

## 2. Authors

- **Gabriel Paz**
- **Diego Linares**
- **Christian Echeverría**

---

## 3. Repository Contents

- `colab-distintas-pruebas.ipynb`  
  Google Colab notebook with:
  - Environment setup (Diffusers, PEFT, CLIP, etc.)
  - Loading the base model (`waifu-diffusion/wd-1-5-beta2`)
  - Loading the trained LoRA weights
  - Visual comparisons (Base vs LoRA)
  - CLIP score experiments and plots

- `pytorch_lora_weights.safetensors`  
  Final LoRA adapter (`lora_ghibli_FINAL`) used for inference.

- *(Optional, if added later)* `data/`  
  Folder containing the curated Ghibli-style dataset.

---

## 4. Dataset & Training Setup

### 4.1 Base Model

We used the anime-oriented Stable Diffusion 1.5 checkpoint:

- `waifu-diffusion/wd-1-5-beta2`

### 4.2 Final Training Configuration

- **Dataset size:** 150 curated images  
  - ~100 close-up portraits  
  - ~50 landscapes / wide shots  
- **Resolution:** 512×512  
- **Training steps:** 1500  
- **LoRA rank:** 16  
- **Learning rate:** `1e-4`  
- **Text encoder:** frozen (`train_text_encoder = False`) so the LoRA focuses on visual style instead of relearning semantics.

### 4.3 Hardware

- Google Colab (NVIDIA T4, 16 GB VRAM)  
- `gradient_checkpointing` was enabled to fit LoRA rank 16 into memory.

---

## 5. Experiments & Comparisons

This section documents the different comparisons we ran in the notebook (`colab-distintas-pruebas.ipynb`). In the written report we only showed one of them in detail; here we list all the relevant experiments as **evidence of the tests performed**.

### 5.1 Ablation: Unbalanced vs Curated Dataset

We ran two main dataset configurations:

1. **V1 – Unbalanced dataset (failure case)**  
   - ~50 images, ~90% landscapes and very few close-up faces.  
   - **Result:**  
     - The model mainly learned **background textures** (grass, clouds, skies).  
     - Faces were not consistently stylized in the Ghibli way.  
     - Outputs often looked like generic anime faces pasted on Ghibli-like backgrounds.

2. **V2 – Curated & balanced dataset (final model)**  
   - 150 images: ~100 portraits + 50 landscapes.  
   - **Result:**  
     - The model successfully applied the Ghibli watercolor style to **characters and faces**.  
     - Proportions and shading were more consistent and recognizable as “Ghibli-like”.  

**Key takeaway:** balancing the dataset (portraits vs landscapes) mattered more than simply adding more images.

---

### 5.2 Visual Comparison: Base Model vs LoRA (Luffy Example)

In the notebook we include side‑by‑side comparisons between:

- **Left:** Base model (`waifu-diffusion/wd-1-5-beta2`)  
- **Right:** Our LoRA (`lora_ghibli_FINAL`)

We use a fixed seed and a Ghibli-style prompt, for example:

```text
ghibli_style, (watercolor:1.2), Luffy from One Piece, smiling, blue sky background
```

**Observations:**

- The **base model** tends to produce a sharp, generic digital anime style.  
- The **LoRA** version shows:
  - Washed-out watercolor colors  
  - Softer shading  
  - Facial proportions and linework closer to Studio Ghibli screenshots  

This demonstrates that the LoRA is not just memorizing images but **overriding the base style** toward the target distribution.

---

### 5.3 Quantitative Comparison: CLIP Score

To go beyond visual inspection, we computed **CLIP scores** to measure semantic alignment between generated images and a reference text prompt, such as:

> `"anime style screenshot from a Studio Ghibli movie, watercolor style"`

Using `openai/clip-vit-base-patch32`, we compared the average CLIP score over a set of generations for:

- **Base model** (no LoRA)  
- **Final LoRA model**

Example result (approximate values):

- **Base model:** CLIP score ≈ **30.20**  
- **LoRA (final):** CLIP score ≈ **31.99**  

The **+1.79 increase** suggests that the LoRA systematically pushes the generated images closer to the “Ghibli watercolor screenshot” concept in CLIP’s latent space.

In the notebook we also include a small bar plot:

- X-axis: `["Base Model", "LoRA Final (Ghibli)"]`  
- Y-axis: CLIP score for each configuration  
- Bars annotated with the numeric values.

---

### 5.4 Prompt Engineering: LoRA vs LoRA + Watercolor Token

Another set of comparisons explores how **prompt engineering interacts with the LoRA**. Our hypothesis is that the LoRA behaves as a **latent enabler**: by itself it shifts the model toward the style, but it works best when combined with the right prompt tokens.

We compared:

1. **LoRA without extra style tokens**  
   - Prompts only referencing `ghibli_style` sometimes still produced more generic anime results.  

2. **LoRA + `(watercolor:1.2)` and related style hints**  
   - Emphasizing watercolor and painting-related tags improved consistency.  
   - Results had:
     - Less “digital” shading  
     - More cohesive Ghibli-like color palettes and textures  

**Conclusion:** the best generations came from **combining the LoRA weights with style-aware prompts**, not from the LoRA alone.

---

## 6. How to Reproduce the Experiments

1. **Clone the repository**

```bash
git clone https://github.com/<your-username>/ghibli-ia-generator-model.git
cd ghibli-ia-generator-model
```

2. **Open the notebook in Google Colab**

- Upload `colab-distintas-pruebas.ipynb` to Colab or open it directly from GitHub.

3. **Mount Google Drive & Model Weights**

- Update the paths in the notebook to point to your Google Drive folder containing:
  - The LoRA weights (`lora_ghibli_FINAL` / `pytorch_lora_weights.safetensors`)
  - The curated dataset (if you want to retrain).

4. **Run the sections**

- **Environment setup:** installs `diffusers`, `peft`, `transformers`, etc.  
- **Visual comparisons:** runs the Base vs LoRA generations.  
- **CLIP experiments:** computes and plots CLIP scores for Base vs LoRA.

---

## 7. Ethical Considerations

This LoRA was trained using **copyrighted material from Studio Ghibli** and is intended **only for academic and research purposes**.

- No commercial use is intended or endorsed.  
- If you build upon this work, please:
  - Respect the rights of original artists.  
  - Follow fair use and local copyright regulations.

---

## 8. Reference Article

For a more detailed narrative of the project, motivations, and lessons learned, see the accompanying article:

> **Teaching AI to Paint Like Ghibli: Why Data Curation Beat Hyperparameter Tuning**  
> *Gabriel Paz, Diego Linares & Christian Echeverría*

You can cite this README and the article as the main documentation of our experiments.
