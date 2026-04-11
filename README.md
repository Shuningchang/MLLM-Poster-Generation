# MLLM-Poster-Generation
Extract the color palette, layout, and overall design mood from the poster and use them in the new product poster generation.


## Environment Details

### Local Environment (Prompt Generation)
- **OS**: Windows 11
- **Python**: 3.14
- **CPU**: AMD Ryzen 5 5625U with Radeon Graphics
- **GPU**: AMD Radeon Graphics (not used, inference runs on CPU)
- **RAM**: 16 GB

### Cloud Environment (Image Generation & Metric Evaluation)
- **Platform**: Google Colab
- **GPU**: NVIDIA T4 (Colab free tier)
- **CUDA**: 11.8

### Required Packages

**Local:**
```bash
pip install torch transformers pillow pandas accelerate
```

**Colab (Image Generation):**
```bash
pip install diffusers==0.24.0 transformers==4.37.0 accelerate==0.26.0 safetensors
```

**Colab (Metric Evaluation):**
```bash
pip install torchmetrics[multimodal] torch-fidelity transformers
pip install git+https://github.com/openai/CLIP.git
```

### Models Used
- **MLLM**: `Qwen/Qwen3-VL-2B-Instruct`
- **T2I**: `runwayml/stable-diffusion-v1-5` + `h94/IP-Adapter` (`ip-adapter_sd15.bin`)
- **Metric**: `openai/clip-vit-base-patch16`

---

## How to Run

### Directory Structure

Please ensure the following structure before running:
task1/
├── MLLM.py              
├── pairs.csv
├── product/
│   ├── p001.png
│   └── ...
└── ref/
├── r001.png
└── ...

---

### Step 1 | Generate Prompts (Local)

Run the following command on your local machine:
```bash
python MLLM.py
```

- The script reads `pairs.csv` and uses Qwen3-VL-2B-Instruct to generate a Stable Diffusion prompt for each product + reference poster pair
- Make sure `DEBUG_MODE = False` in the script to process all 100 pairs
- Inference runs on **CPU**; estimated runtime is approximately 60–120 minutes
- Output: `prompts.json`

---

### Step 2 | Generate Poster Images (Google Colab)

1. Compress and upload the following files to Colab:
   - `prompts.json`
   - `product/` (compressed as `product.zip`)
   - `ref/` (compressed as `ref.zip`)
   - `pairs.csv`

2. Open `task1_image_generation.ipynb` in Colab and run all cells in order

3. Image generation parameters:
   - `num_inference_steps = 30`
   - `guidance_scale = 8.0`
   - `seed = 456`
   - IP-Adapter scale = 0.6

4. All generated images are automatically resized to **224×224** and named by `pair_id` (e.g., `0001.jpg`)

5. Run the packaging cell to download `generated_images.zip`

---

### Step 3 | Evaluate Metrics (Google Colab)

Open `task1_metrics.ipynb` in Colab and run all cells in order. The following three metrics will be computed:

| Metric | Description |
|--------|-------------|
| CLIP-Score | Semantic similarity between generated poster and editing prompt |
| CLIP Visual Similarity | Visual similarity between generated poster and product image |
| KID | Distribution distance between generated posters and reference posters (lower is better) |

---

### Output Format
task1_images/
├── 0001.jpg   (224×224)
├── 0002.jpg   (224×224)
└── ...        (100 in total)
