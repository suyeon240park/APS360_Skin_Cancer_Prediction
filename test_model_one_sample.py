import torch
import torchvision.transforms as transforms
import torchvision.models as models
import torch.nn as nn
from PIL import Image
import os

# Model architecture (same as during training)
class SkinLesionClassifier(nn.Module):
    def __init__(self, num_classes=4):
        super(SkinLesionClassifier, self).__init__()
        # Use ResNet50 as backbone
        self.backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        
        # Remove the fully connected layer and replace with GAP
        num_ftrs = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()
        
        # Global Average Pooling
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Fully connected classifier
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(num_ftrs, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes),
            nn.Softmax(dim=1)  # Softmax for classification
        )

    def forward(self, x):
        features = self.backbone(x)  # Extract features
        features = self.global_avg_pool(features)  # Apply GAP
        return self.classifier(features)

# Load the best model
def load_model(model_path, num_classes=4):
    model = SkinLesionClassifier(num_classes=num_classes)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

# Define the transformation for new data
def get_transforms():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])

# Load and preprocess a single image
def load_single_image(image_path):
    # Load image using PIL
    image = Image.open(image_path).convert('RGB')
    
    # Apply transformations
    transform = get_transforms()
    image_tensor = transform(image)
    
    # Add batch dimension
    image_tensor = image_tensor.unsqueeze(0)
    
    return image_tensor

# Make prediction for a single image
def predict_single_image(model, image_tensor):
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = outputs[0]
        predicted_class = torch.argmax(probabilities).item()
        confidence = probabilities[predicted_class].item()
    return predicted_class, confidence

# Example usage
if __name__ == "__main__":
    # Get the directory where the script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Create absolute paths for model and image
    model_path = os.path.join(script_dir, 'best_model.pth')
    image_path = os.path.join(script_dir, 'test_data.png')
    
    # Load model and image
    model = load_model(model_path)
    image_tensor = load_single_image(image_path)
    
    # Make prediction
    predicted_class, confidence = predict_single_image(model, image_tensor)
    
    # Print results
    class_names = ['bcc', 'benign', 'melanoma', 'scc']
    print(f"Predicted class: {class_names[predicted_class]}")
    print(f"Confidence: {confidence:.2%}")
