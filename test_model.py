import torch
import torchvision.transforms as transforms
from torchvision import datasets
from torch.utils.data import DataLoader
import torchvision.models as models
import torch.nn as nn

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

# Load new data
def load_new_data(data_path, batch_size=32):
    dataset = datasets.ImageFolder(root=data_path, transform=get_transforms())
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    return data_loader

# Make predictions
def predict(model, data_loader):
    all_preds = []
    with torch.no_grad():
        for inputs, _ in data_loader:
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy())
    return all_preds

# Example usage
if __name__ == "__main__":
    model_path = 'best_model.pth'
    new_data_path = 'test_data.jpg'
    batch_size = 32

    model = load_model(model_path)
    new_data_loader = load_new_data(new_data_path, batch_size)
    predictions = predict(model, new_data_loader)

    print("Predictions:", predictions)