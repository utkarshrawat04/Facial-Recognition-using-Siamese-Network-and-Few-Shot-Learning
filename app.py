import streamlit as st
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import os


class ResNetSiamese(nn.Module):
    def __init__(self):
        super().__init__()
        resnet = models.resnet18(pretrained=True)
        self.feature_extractor = nn.Sequential(*list(resnet.children())[:-1])
        self.fc = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(), nn.Dropout(0.7),
            nn.Linear(512, 256),
            nn.ReLU(), nn.Dropout(0.6),
            nn.Linear(256, 128),
            nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(128, 1)
        )

    def forward_one(self, x):
        x = self.feature_extractor(x)
        return x.view(x.size(0), -1)

    def forward(self, x1, x2):
        f1, f2 = self.forward_one(x1), self.forward_one(x2)
        dist = torch.abs(f1 - f2)
        return self.fc(dist)


@st.cache_resource
def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ResNetSiamese().to(device)
    model.load_state_dict(torch.load(r"C:\Users\athar\Downloads\EC_ATML_Project\Siamese_VGG_83.pth", map_location=device))
    model.eval()
    return model


def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    return transform(image).unsqueeze(0)


def weighted_average(distances, epsilon=1e-6):
    d_max = max(distances)
    weights = [(d_max - d + epsilon) for d in distances]
    return sum(w * d for w, d in zip(weights, distances)) / sum(weights)


def main():
    st.title("Face Recognition System")
    st.write("Siamese Network with ResNet-18 Backbone")

    model = load_model()
    app_mode = st.sidebar.selectbox("Select Mode", ["Support Set Recognition", "Pairwise Comparison"])

    if app_mode == "Support Set Recognition":
        handle_support_set_mode(model)
    else:
        handle_pairwise_mode(model)

def handle_support_set_mode(model):
    st.sidebar.subheader("Support Set Settings")
    threshold = st.sidebar.slider("Threshold", 0.0, 10.0, 6.0, 0.1)
    support_dir = st.sidebar.text_input("Support Directory", r"C:\Path\To\SupportSet")
    uploaded_file = st.file_uploader("Upload Query Image", type=["jpg", "jpeg", "png"])

    if uploaded_file and os.path.exists(support_dir):
        query_image = Image.open(uploaded_file).convert("RGB")
        st.image(query_image, caption="Query Image", width=300)
        best_match, min_dist, all_distances = process_support_set(query_image, model, support_dir)

        st.subheader("Recognition Result")
        if best_match and min_dist <= threshold:
            st.success(f" Match: {best_match} (Distance: {min_dist:.2f})")
        else:
            st.error(f" No Match (Best Distance: {min_dist:.2f})")

        st.subheader("All Distances")
        for person, dist in all_distances:
            st.write(f"{person}: {dist:.4f}")

def handle_pairwise_mode(model):
    st.subheader("Pairwise Image Comparison")
    threshold = st.sidebar.slider("Similarity Threshold", 0.0, 1.0, 0.6, 0.01)

    col1, col2 = st.columns(2)
    with col1:
        img1 = st.file_uploader("Upload Image 1", type=["jpg", "jpeg", "png"], key="img1")
    with col2:
        img2 = st.file_uploader("Upload Image 2", type=["jpg", "jpeg", "png"], key="img2")

    if img1 and img2:
        image1 = Image.open(img1).convert("RGB")
        image2 = Image.open(img2).convert("RGB")

        st.image(image1, caption="Image 1", use_column_width=True)
        st.image(image2, caption="Image 2", use_column_width=True)

        if st.button("Compare Images"):
            score, is_same = compare_images(model, image1, image2, threshold)
            st.subheader("Result")
            st.metric("Similarity Score", f"{score:.4f}")
            if is_same:
                st.success(" Same Identity")
            else:
                st.error(" Different Identities")


def compare_images(model, image1, image2, threshold):
    img1_tensor = preprocess_image(image1).to(next(model.parameters()).device)
    img2_tensor = preprocess_image(image2).to(next(model.parameters()).device)

    with torch.no_grad():
        output = model(img1_tensor, img2_tensor)
        prob = torch.sigmoid(output).item()

    return prob, prob > threshold

def process_support_set(query_image, model, support_dir):
    query_tensor = preprocess_image(query_image).to(next(model.parameters()).device)
    best_match = None
    min_distance = float('inf')
    all_distances = []

    for person in os.listdir(support_dir):
        person_dir = os.path.join(support_dir, person)
        if not os.path.isdir(person_dir):
            continue

        distances = []
        for img_file in os.listdir(person_dir)[:5]:
            try:
                support_img = Image.open(os.path.join(person_dir, img_file)).convert("RGB")
                support_tensor = preprocess_image(support_img).to(next(model.parameters()).device)
                distance = torch.norm(model.forward_one(query_tensor) - model.forward_one(support_tensor)).item()
                distances.append(distance)
            except:
                continue

        if distances:
            avg_dist = weighted_average(distances)
            all_distances.append((person, avg_dist))
            if avg_dist < min_distance:
                min_distance = avg_dist
                best_match = person

    all_distances.sort(key=lambda x: x[1])
    return best_match, min_distance, all_distances


if __name__ == "__main__":
    main()
