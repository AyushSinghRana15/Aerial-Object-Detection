import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import streamlit as st
from PIL import Image
import torchvision.transforms as transforms
from torchvision.models import ResNet50_Weights, MobileNet_V2_Weights, EfficientNet_B0_Weights

# CustomCNN class
class CustomCNN(nn.Module):
    def __init__(self):
        super(CustomCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout1 = nn.Dropout(0.25)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.dropout2 = nn.Dropout(0.25)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.dropout3 = nn.Dropout(0.25)
        self.fc1 = nn.Linear(128 * 28 * 28, 512)
        self.dropout_fc = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, 2)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.dropout1(x)
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.dropout2(x)
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = self.dropout3(x)
        x = x.view(-1, 128 * 28 * 28)
        x = F.relu(self.fc1(x))
        x = self.dropout_fc(x)
        x = self.fc2(x)
        return x

# Cache to load models once
@st.cache_resource
def load_models():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Helper function to safely load checkpoint
    def safe_load_checkpoint(model, checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model_dict = model.state_dict()
        filtered_dict = {k: v for k, v in checkpoint.items() if k in model_dict and v.size() == model_dict[k].size()}
        model_dict.update(filtered_dict)
        model.load_state_dict(model_dict)
        return model

    # ResNet50
    weights_resnet = ResNet50_Weights.DEFAULT
    model_resnet = torchvision.models.resnet50(weights=weights_resnet)
    num_ftrs = model_resnet.fc.in_features
    model_resnet.fc = nn.Linear(num_ftrs, 2)
    model_resnet = safe_load_checkpoint(model_resnet, 'resnet50_best.pth')
    model_resnet.to(device)
    model_resnet.eval()

    # MobileNetV2
    weights_mobilenet = MobileNet_V2_Weights.DEFAULT
    model_mobilenet = torchvision.models.mobilenet_v2(weights=weights_mobilenet)
    in_features = model_mobilenet.classifier[1].in_features
    model_mobilenet.classifier[1] = nn.Linear(in_features, 2)
    model_mobilenet = safe_load_checkpoint(model_mobilenet, 'mobilenet_best.pth')
    model_mobilenet.to(device)
    model_mobilenet.eval()

    # EfficientNetB0
    weights_efficientnet = EfficientNet_B0_Weights.DEFAULT
    model_efficient = torchvision.models.efficientnet_b0(weights=weights_efficientnet)
    in_features = model_efficient.classifier[1].in_features
    model_efficient.classifier[1] = nn.Linear(in_features, 2)
    model_efficient = safe_load_checkpoint(model_efficient, 'efficientnet_b0_best.pth')
    model_efficient.to(device)
    model_efficient.eval()

    # Custom CNN
    model_cnn = CustomCNN()
    model_cnn = safe_load_checkpoint(model_cnn, 'best_model.pth')
    model_cnn.to(device)
    model_cnn.eval()

    return {
        "ResNet50": model_resnet,
        "MobileNetV2": model_mobilenet,
        "EfficientNetB0": model_efficient,
        "Custom CNN": model_cnn
    }, device

models, device = load_models()

st.title("🖼️ Aerial Image Classification App")

model_name = st.selectbox(
    "Select model:",
    list(models.keys())
)

# Display warning if user selects model other than ResNet50
if model_name != "ResNet50":
    st.caption("⚠️ **Warning:** Selected model is less accurate compared to ResNet50 (98% accuracy).")

# Replace your image processing section with this:
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
camera_image = st.camera_input("Or capture an image using your camera")

image = None
source = None

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    source = "uploaded"
    st.image(image, caption="Uploaded Image", use_column_width=True)
elif camera_image is not None:
    image = Image.open(camera_image).convert("RGB")
    source = "captured"
    st.image(image, caption="Captured Image", use_column_width=True)

if image is not None:
    # Define transform here (same as training)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    
    input_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = models[model_name](input_tensor)
        probs = torch.nn.functional.softmax(output, dim=1)
        conf, pred = torch.max(probs, 1)

    class_names = ["bird", "drone"]
    
    col1, col2 = st.columns(2)
    with col1:
        st.success(f"**Prediction:** {class_names[pred.item()]}")
    with col2:
        st.metric("Confidence", f"{conf.item()*100:.1f}%")
    
    # Show probability distribution
    st.write("**Probability distribution:**")
    prob_dict = {class_names[i]: probs[0][i].item() for i in range(2)}
    st.bar_chart(prob_dict)

    
    # Show probability distribution
    st.write("**Probability distribution:**")
    prob_dict = {class_names[i]: probs[0][i].item() for i in range(2)}
    st.bar_chart(prob_dict)
