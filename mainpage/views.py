from django.shortcuts import render, redirect
from django.contrib.auth.forms import AuthenticationForm, UserCreationForm
from django.contrib.auth import authenticate, login
from django.contrib import messages
from django.core.files.storage import FileSystemStorage
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np
import os
from django.conf import settings

# Define the model
class SkinDiseaseClassifier(nn.Module):
    def __init__(self, num_classes):
        super(SkinDiseaseClassifier, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(32 * 62 * 62, 64)
        self.fc2 = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Load model
model_path = os.path.join(settings.BASE_DIR, 'skin_disease_model.pth')
num_classes = 8  # Update based on your dataset
model = SkinDiseaseClassifier(num_classes=num_classes)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.load_state_dict(torch.load(model_path, map_location=device))
model = model.to(device)
model.eval()

# Main page view
def mainpage(request):
    return render(request, 'mainpage.html')

# Login page view
def loginpage(request):
    if request.method == "POST":
        if request.user.is_authenticated:
            return redirect('/profilepage')
        form = AuthenticationForm(data=request.POST)
        if form.is_valid():
            username = form.cleaned_data['username']
            password = form.cleaned_data['password']
            user = authenticate(request, username=username, password=password)
            if user is not None:
                login(request, user)
                return redirect('/profilepage')
            else:
                messages.error(request, 'Invalid Username/Password')
        else:
            messages.error(request, 'Invalid form submission')
    form = AuthenticationForm()
    return render(request, 'loginpage.html', {'form': form})

# Signup page view
def signuppage(request):
    if request.method == "POST":
        form = UserCreationForm(request.POST)
        if form.is_valid():
            form.save()
            username = form.cleaned_data['username']
            password = form.cleaned_data['password1']
            user = authenticate(username=username, password=password)
            if user is not None:
                login(request, user)
                return redirect('/profilepage')
        else:
            messages.error(request, 'User registration failed. Please check your input.')
    else:
        form = UserCreationForm()
    return render(request, 'signuppage.html', {'form': form})

# Profile page view for image analysis
def profilepage(request):
    if request.user.is_authenticated:
        if request.method == "POST":
            if request.FILES.get('uploadImage'):
                img_name = request.FILES['uploadImage']
                fs = FileSystemStorage()
                filename = fs.save(img_name.name, img_name)
                img_url = fs.url(filename)
                img_path = fs.path(filename)

                # Load and preprocess the image
                img = cv2.imread(img_path, cv2.IMREAD_COLOR)
                if img is None:
                    messages.error(request, "Error reading the image file. Please upload a valid image.")
                    return render(request, 'profilepage.html')

                img = cv2.resize(img, (250, 250))
                img = img / 255.0  # Normalize
                img = torch.tensor(img, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0).to(device)

                # Make prediction
                try:
                    with torch.no_grad():
                        outputs = model(img)
                        _, predicted = torch.max(outputs, 1)
                        predict = predicted.item()
                except Exception as e:
                    messages.error(request, f"Model prediction error: {str(e)}")
                    return render(request, 'profilepage.html', {'message': "Error in prediction."})

                # Disease classification
                skin_disease_names = [
                    'Cellulitis', 'Impetigo', 'Athlete Foot', 'Nail Fungus',
                    'Ringworm', 'Cutaneous Larva Migrans', 'Chickenpox', 'Shingles'
                ]
                diagnosis = ['']  # Add descriptions if available

                result1 = skin_disease_names[predict] if 0 <= predict < len(skin_disease_names) else "Unknown disease"
                result2 = diagnosis[predict] if len(diagnosis) > predict else "Diagnosis not available"

                return render(request, 'profilepage.html', {'img': img_url, 'obj1': result1, 'obj2': result2})
            else:
                messages.error(request, "Please select an image to upload.")
        return render(request, 'profilepage.html')
    return redirect("/loginpage")

# About page view
def about(request):
    return render(request, 'about.html')
