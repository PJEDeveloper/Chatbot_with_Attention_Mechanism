# Chatbot with Attention Mechanism

# 🗣️ Chatbot with Attention Mechanism using LSTM

This project implements a bilingual chatbot trained on the TED Talks translation dataset (Portuguese-to-English). The architecture is built on an encoder-decoder LSTM with a custom **cross-attention mechanism** and supports both **GPU-based training** and **inference** using TensorFlow.

---

## 🔍 Overview

This chatbot demonstrates the use of attention-based sequence-to-sequence models for neural machine translation (NMT). It leverages TensorFlow’s deep learning capabilities and includes training, evaluation, and real-time inference scripts.

## 🚀 Features

- Encoder-Decoder LSTM architecture
- Custom cross-attention layer
- Tokenizer with start/end token encoding
- TED Talks translation dataset via `tensorflow_datasets`
- GPU acceleration with memory growth
- Model persistence: weights + tokenizer
- Configurable hyperparameters

---

## 🛠️ Setup & Installation

```bash
git clone https://github.com/PJEDeveloper/chatbot-with-attention.git
cd chatbot-with-attention
python -m venv chatbot_env
source chatbot_env/bin/activate
pip install -r requirements.txt
```

---

## 🏋️‍♂️ Training the Model

Run the training script with or without validation/metrics logging:

### Standard Training
```bash
python wsl2_lstm_atten_chatbot_gpu_train_v1.py
```

### With Validation & Early Stopping
```bash
python wsl2_lstm_atten_chatbot_gpu_train_v2_metrics.py
```

This will:
- Download and preprocess the `ted_hrlr_translate/pt_to_en` dataset
- Build vocabulary and tokenizer
- Train over 10–150 epochs with attention-enhanced decoding
- Save:
  - `encoder.weights.h5`
  - `decoder.weights.h5`
  - `tokenizer.pkl`  
  to `model_results/`

---

## 🤖 Running Inference

Load the trained model and generate responses interactively:
```bash
python wsl2_lstm_atten_chatbot_gpu_inference_v1.py
```

Edit the `input_sentence` variable to test different queries:
```python
input_sentence = "Hello, how are you?"
```

The script will:
- Load `tokenizer.pkl`
- Tokenize & pad input
- Use attention-enhanced decoding to produce output
- Display: `Generated Response: <your_bot_reply_here>`

---

## 📂 Files in This Repo

| File | Description |
|------|-------------|
| `wsl2_lstm_atten_chatbot_gpu_train_v1.py` | Basic training script |
| `wsl2_lstm_atten_chatbot_gpu_train_v2_metrics.py` | Enhanced training with validation and early stopping |
| `wsl2_lstm_atten_chatbot_gpu_inference_v1.py` | Inference pipeline using saved models |
| `requirements.txt` | Dependency list |
| `model_results/` | Saved weights and tokenizer |

---

## 📊 Dataset

- [TensorFlow Datasets - TED HRLR pt_to_en](https://www.tensorflow.org/datasets/catalog/ted_hrlr_translate)
- ~36k sentence pairs (train), 1k (validation)
- Automatically downloaded via `tfds.load`

---

## 📝 License

This project is licensed under the Apache 2.0 License.

---

> Developed by Patrick Hill — AI Developer and U.S. Air Force Veteran  
> [LinkedIn](https://www.linkedin.com/in/patrick-hill-4b9807178/)
