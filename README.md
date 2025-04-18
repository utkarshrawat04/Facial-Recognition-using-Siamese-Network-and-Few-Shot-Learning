# 🧠 Facial Recognition using Siamese Network and Few-Shot Learning (with PSO & GANs)

This project implements a **Facial Recognition System** using a **Siamese Neural Network** enhanced with **Few-Shot Learning**, **Generative Adversarial Networks (GANs)** for data augmentation, and **Particle Swarm Optimization (PSO)** for hyperparameter tuning. The project is designed to work even with very limited samples per identity and includes a user-friendly **Streamlit web app** for real-time face verification.

---

## 👨‍💻 Authors
- Atharva Pande : https://github.com/yoboi1234673
- Tanish Punamiya : https://github.com/Tanish1302  
- Utkarsh Rawat  : https://github.com/utkarshrawat04

---

## 🛠️ Features
- Siamese Network with **ResNet-18 backbone**
- Few-shot image pair generation for training
- **Binary classification using BCE Loss**
- **GAN-based augmentation** for the support set
- **PSO** for tuning thresholds and augmentation parameters
- **Streamlit GUI** for user interaction with pairwise and support set recognition modes
- Evaluation metrics: **Accuracy, ROC-AUC, Confusion Matrix**

---

## 📁 Project Structure

```
📂 Facial-Recognition-using-Siamese-Network-and-Few-Shot-Learning/
├── siamese-network-vggface2.ipynb       # Training and evaluation code
├── app.py                               # Streamlit app for verification
├── Siamese_VGG_83.pth                   # Trained model weights (load locally)
├── README.md                            # You're here!
└── GAN                                  # GAN Folder
```

---

## 📦 Installation & Setup

1. **Clone the repository**:
```bash
git clone https://github.com/utkarshrawat04/Facial-Recognition-using-Siamese-Network-and-Few-Shot-Learning.git
cd Facial-Recognition-using-Siamese-Network-and-Few-Shot-Learning
```

2. **Create a virtual environment** *(optional but recommended)*:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**:
```bash
pip install torch torchvision streamlit Pillow
```

4. **Place model weights**  
Make sure `Siamese_VGG_83.pth` is available at the correct path as mentioned in `app.py`.

---

## ▶️ Running the App

Launch the Streamlit app:
```bash
streamlit run app.py
```

### Modes:
- **Support Set Recognition**  
  Upload a query image and compare it against multiple folders of identities.
  
- **Pairwise Comparison**  
  Upload any two images to check if they belong to the same identity.

---

## 🖼️ Support Set Directory Format

```
support_set/
├── person1/
│   ├── img1.jpg
│   ├── img2.jpg
├── person2/
│   ├── img1.jpg
│   ├── img2.jpg
...
```



## 🔍 Example Output

### Support Set Recognition
![WhatsApp Image 2025-04-15 at 2 49 33 PM](https://github.com/user-attachments/assets/1f7cb456-2aab-42e5-bd90-637c6f19b008)

### Pairwise Comparison
![WhatsApp Image 2025-04-15 at 2 51 08 PM](https://github.com/user-attachments/assets/9f99b2ea-b484-43fb-8f97-cd49a6d1ae27)
![WhatsApp Image 2025-04-15 at 2 51 39 PM](https://github.com/user-attachments/assets/5e526298-611a-4140-a0b3-faaf414b5fa0)
![WhatsApp Image 2025-04-15 at 2 52 10 PM](https://github.com/user-attachments/assets/e83d70f2-79a2-420d-878b-f8ef22d50a8a)
![WhatsApp Image 2025-04-15 at 2 52 25 PM](https://github.com/user-attachments/assets/230eac61-523a-4d46-9dc9-16083e373718)





---

## 📊 Evaluation Results

- **Validation Accuracy**: 85%
![WhatsApp Image 2025-04-15 at 2 40 50 PM](https://github.com/user-attachments/assets/be252667-64b3-4270-8db1-da6e50d96a5e)


  
- **ROC-AUC Score**: 0.91
  
![WhatsApp Image 2025-04-15 at 2 40 36 PM](https://github.com/user-attachments/assets/d075365a-d3d3-4a88-881e-42b9209713bf)


- **Confusion Matrix**:
  - True Positives: 4210  
  - False Negatives: 819  
  - False Positives: 816  
  - True Negatives: 4135
 
    
![WhatsApp Image 2025-04-15 at 2 40 45 PM](https://github.com/user-attachments/assets/0ba79003-580b-41d6-b64d-077d38ac846e)

---

## 🧬 Role of GANs in Facial Recognition

Generative Adversarial Networks (GANs) are a powerful class of deep generative models that consist of two competing networks: a **Generator** and a **Discriminator**. While GANs are not the core component in our current Siamese Network-based system, they hold substantial potential for augmenting facial recognition tasks — especially under **few-shot learning** constraints.

### 🔄 CycleGAN and PatchGAN in This Context

- **CycleGAN** can be used for **unpaired image-to-image translation**, which enables generating new facial expressions, angles, or lighting conditions from limited identity samples — without needing paired training data.
  
- **PatchGAN** focuses on classifying whether *patches* of an image are real or fake rather than the entire image, making it particularly useful for **fine-grained feature generation**. This is ideal for improving facial texture quality in augmented datasets.

### 🚀 Future Work with GANs

In future iterations of this project, GANs could be leveraged for:
- **Expanding support sets** with synthetic variations for better generalization
- Generating realistic but unseen facial expressions or lighting to enrich training pairs
- Improving recognition robustness under domain shifts (e.g., different cameras or lighting)
- Data augmentation in scenarios where data privacy restricts real image collection

Integrating CycleGAN or PatchGAN can significantly enhance the diversity and realism of training data, which in turn improves the performance of verification models under constrained-data regimes.

---





