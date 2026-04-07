# MLLM-Poster-Generation
Extract the color palette, layout, and overall design mood from the poster and use them in the new product poster generation.

### 環境說明 (Environment Details)

### 本機環境（Prompt 生成）
- **作業系統**：Windows 11
- **Python**：3.14
- **CPU**：AMD Ryzen 5 5625U with Radeon Graphics
- **RAM**：16 GB

### 雲端環境（圖片生成 & Metric 計算）
- **平台**：Google Colab
- **GPU**：NVIDIA T4（Colab 免費版）
- **CUDA**：11.8

### 使用套件

**本機：**
```bash
pip install torch transformers pillow pandas accelerate
```

**Colab（圖片生成）：**
```bash
pip install diffusers==0.24.0 transformers==4.37.0 accelerate==0.26.0 safetensors
```

**Colab（Metric 計算）：**
```bash
pip install torchmetrics[multimodal] torch-fidelity transformers
pip install git+https://github.com/openai/CLIP.git
```

### 使用模型
- **MLLM**: `Qwen/Qwen3-VL-2B-Instruct`
- **T2I**: `runwayml/stable-diffusion-v1-5` + `h94/IP-Adapter` (`ip-adapter_sd15.bin`)
- **CLIP-Score**: `openai/clip-vit-base-patch16`（via Hugging Face transformers）
- **CLIP Visual Similarity**: `openai/ViT-B/16`（via openai/CLIP）
- **KID**: Inception-v3（torch-fidelity 內建）


---

## 執行方式 (How to Run)

### 資料夾結構

執行前請確認資料夾結構如下：
hw1_<student_id>_task1/
├── ai_hw1.py              ← Prompt 生成主程式
├── pairs.csv
├── product/
│   ├── p001.png
│   └── ...
└── ref/
├── r001.png
└── ...

### Step 1｜生成 Prompts（本機執行）

在本機執行以下指令：
```bash
python ai_hw1.py
```

- 程式會讀取 `pairs.csv`，對每一組 product + ref poster 用 Qwen3-VL-2B-Instruct 生成 Stable Diffusion prompt
- 確認程式內 `DEBUG_MODE = False` 以處理全部 100 筆
- 推理在 **CPU** 上執行，約 180-200 分鐘
- 輸出檔案：`prompts.json`

---

### Step 2｜生成海報圖片（Google Colab 執行）

1. 將以下檔案壓縮並上傳至 Colab：
   - `prompts.json`
   - `product/`（壓縮為 `product.zip`）
   - `ref/`（壓縮為 `ref.zip`）
   - `pairs.csv`

2. 在 Colab 開啟 `task1_image_generation.ipynb`，依序執行所有 Cell

3. 圖片生成參數：
   - `num_inference_steps = 30`
   - `guidance_scale = 8.0`
   - `seed = 456`
   - IP-Adapter scale = 0.6

4. 生成完成後所有圖片會自動 resize 為 **224×224**，並以 `pair_id` 命名（例如 `0001.jpg`）

5. 執行打包 Cell 下載 `generated_images.zip`

---

### Step 3｜計算 Metrics（Google Colab 執行）

在 Colab 開啟 `task1_metrics.ipynb`，依序執行所有 Cell，會計算以下三個指標：

| Metric | 說明 |
|--------|------|
| CLIP-Score | 生成圖與 editing prompt 的語意相似度 |
| CLIP Visual Similarity | 生成圖與 product image 的視覺相似度 |
| KID | 生成圖與 ref poster 的分佈距離（越低越好） |

---

### 輸出格式
hw1_<student_id>_task1_images/
├── 0001.jpg   (224×224)
├── 0002.jpg   (224×224)
└── ...        (共 100 張)
