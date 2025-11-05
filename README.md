# 🧠 Federated xDNN for Interpretable Dental View Classification

This repository implements **Federated Explainable Deep Neural Networks (FxDNN)** — a framework that combines **federated learning** with **explainable prototype-based classification** for dental view image analysis.

The project proposes a privacy-preserving and interpretable model that allows multiple clinical sites to collaboratively train a shared encoder while keeping their local data private. Each client trains its own **xDNN classifier** locally on embeddings extracted from a global encoder. Interpretability is achieved through **prototype inspection** and **SHAP-based visual explanations**.

---

## 🚀 Key Features

- **Federated Encoder Training** — Train a shared image encoder (`ResNet18` or `ResNet50`) across distributed clients using **FedAvg** or **FedProx**.  
- **Local xDNN Classifiers** — Each client learns an **xDNN** model based on **prototypes** extracted from its embeddings, without sharing data.  
- **Explainability via SHAP** — Generates **heatmaps** and **prototype visualizations** that highlight image regions contributing to classification.  
- **Privacy by Design** — No raw data or prototypes are exchanged — only model weights are aggregated.

---

## 🧩 Repository Structure

ProtoPNet+xDNN/
│
├── fed_xdnn.py # Main federated and local training script
├── analyze_prototypes_shap.py # SHAP-based visual interpretation of prototypes
│
├── encoder_global.pt # Federated global encoder (saved model)
├── siteA_xdnn.pkl # Local xDNN model for client A
├── siteB_xdnn.pkl # Local xDNN model for client B
│
├── data/
│ └── clients/
│ ├── siteA/
│ │ ├── train/{Frontal,Lateral,Oclusal}/...
│ │ └── val/{Frontal,Lateral,Oclusal}/...
│ └── siteB/
│ ├── train/{Frontal,Lateral,Oclusal}/...
│ └── val/{Frontal,Lateral,Oclusal}/...
│
├── shap_siteA/ # SHAP visual explanations for siteA
└── shap_siteB/ # SHAP visual explanations for siteB

---

## ⚙️ Installation

```bash
git clone https://github.com/<your-username>/Federated-xDNN.git
cd Federated-xDNN
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
Required packages (if you don’t use requirements.txt):

nginx
Copiar código
torch torchvision numpy scikit-learn scipy shap matplotlib opencv-python
🏗️ Federated Encoder Training
Train a global encoder collaboratively across all client datasets.

bash
Copiar código
python fed_xdnn.py --mode federated \
  --clients_root data/clients \
  --rounds 5 --local_epochs 1 --batch_size 32 --lr 3e-4 \
  --model resnet18 --image_size 224 --fedprox_mu 0.0 \
  --save_encoder encoder_global.pt
This performs FedAvg (or FedProx if --fedprox_mu > 0) to stabilize training with heterogeneous data.

🧠 Local xDNN Training
Each site uses the global encoder to build its local explainable model:

bash
Copiar código
python fed_xdnn.py --mode local_xdnn \
  --clients_root data/clients --client_id siteA \
  --encoder_path encoder_global.pt \
  --batch_size 32 --image_size 224
Outputs:

Training/validation metrics (accuracy, precision, recall, F1, κ)

Prototype counts per class

Saved model: siteA_xdnn.pkl (or siteB_xdnn.pkl)

🔍 SHAP-Based Prototype Explanation
Generate interpretability heatmaps for prototypes and their most similar images.

bash
Copiar código
python analyze_prototypes_shap.py \
  --clients_root data/clients --client_id siteA \
  --encoder_path encoder_global.pt --xdnn_path siteA_xdnn.pkl \
  --split val --per_class 1 --top_per_proto 1 --image_size 224 \
  --out_dir shap_siteA --alpha 0.35 --sigma 0.8 --cmap seismic
Output
Heatmaps overlaid on denormalized images (red = positive influence, blue = negative), illustrating how xDNN prototypes align with image features (e.g., occlusal line, lateral curvature, frontal symmetry).

🧮 Evaluation Metrics
Metric	Description
Accuracy	Overall correctness
Precision / Recall / F1	Class-level performance
Cohen’s κ	Chance-corrected agreement
Confusion Matrix	Prediction distribution per class

📈 Example Results
Client	Accuracy	F1	κ	Notes
Site A	0.89	0.88	0.82	Minor confusion Frontal ↔ Lateral
Site B	1.00	1.00	1.00	Fully converged
Global	–	–	–	Shared encoder generalized well

(Values from a representative run; results vary by seed and preprocessing.)

🧩 Explainability Insight
Red regions: evidence for the predicted class.

Blue regions: evidence against the predicted class.

Prototype visualization confirms that xDNN focuses on semantically meaningful patterns.

📚 Citation
If you use this code or findings, please cite:

bibtex
Copiar código
@article{Souza2025FederatedxDNN,
  title     = {Federated xDNN for Interpretable Dental View Classification},
  author    = {Paulo Vitor de Campos Souza},
  journal   = {IEEE Journal of Translational Engineering in Health and Medicine},
  year      = {2025},
  note      = {Under Review}
}
🙏 Acknowledgment
This work was supported by national funds through the Fundação para a Ciência e a Tecnologia (FCT), under project UIDB/04152 – Centro de Investigação em Gestão de Informação (MagIC), NOVA Information Management School (NOVA IMS), Universidade Nova de Lisboa, Portugal.

We also acknowledge the Brazilian Ministry of Health for providing access to calibration materials and public documentation from the SB Brasil 2023 National Oral Health Survey.

⚖️ License
Released under the MIT License. You are free to use, modify, and distribute this code with proper citation.

📬 Contact
Paulo Vitor de Campos Souza
NOVA Information Management School (NOVA IMS)
Email: psouza@novaims.unl.pt

✳️ “Bridging public health and computer vision for interpretable oral-health AI.”
