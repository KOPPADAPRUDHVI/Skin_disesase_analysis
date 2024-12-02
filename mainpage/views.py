import os
import torch
import timm
from torch import nn
from django.shortcuts import render, redirect
from django.contrib.auth.forms import AuthenticationForm, UserCreationForm
from django.contrib.auth import authenticate, login, logout
from django.contrib import messages
from django.core.files.storage import FileSystemStorage
import cv2
from django.conf import settings

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Define the path to the saved model
MODEL_PATH = r"C:\Users\ACER\Desktop\Infosys Project\dino_skin_disease_model.pth"
num_classes = 8  # Update based on your dataset

# Load the Vision Transformer model from timm
def load_model():
    model = timm.create_model('vit_base_patch16_224', pretrained=True)  # Vision Transformer model from timm
    model.head = nn.Linear(model.head.in_features, num_classes)  # Modify final layer to match number of classes
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))  # Load model weights
    model = model.to(device)  # Ensure model is on correct device (GPU or CPU)
    model.eval()  # Set the model to evaluation mode
    return model

# Load the model once when the server starts
model = load_model()

# Main page view
def mainpage(request):
    return render(request, 'mainpage.html')



def loginpage(request):
    if 'messages' in request.session:
        del request.session['messages']

    if request.method == "POST":
        form = AuthenticationForm(data=request.POST)
        if form.is_valid():
            username = form.cleaned_data['username']
            password = form.cleaned_data['password']
            user = authenticate(request, username=username, password=password)
            if user is not None:
                login(request, user)
                return redirect('/profilepage')
            else:
                messages.error(request, 'Invalid Username or Password')
        else:
            # Aggregate all errors into a single message
            error_messages = []
            for field, errors in form.errors.items():
                error_messages.extend(errors)
            messages.error(request, " ".join(error_messages))
    else:
        form = AuthenticationForm()

    return render(request, 'loginpage.html', {'form': form})




def signuppage(request):
    if request.method == "POST":
        form = UserCreationForm(request.POST)
        if form.is_valid():
            user = form.save()
            username = form.cleaned_data['username']
            password = form.cleaned_data['password1']
            user = authenticate(username=username, password=password)
            if user is not None:
                login(request, user)
                return redirect('/profilepage')
        else:
            # Aggregate all errors into a single message
            error_messages = []
            for field, errors in form.errors.items():
                error_messages.extend(errors)
            messages.error(request, " ".join(error_messages))
    else:
        form = UserCreationForm()
    return render(request, 'signuppage.html', {'form': form})



# Profile page view
def profilepage(request):
    if not request.user.is_authenticated:  # Ensure the user is logged in
        return redirect('/loginpage')

    context = {}
    
    if request.method == "POST":
        uploaded_file = request.FILES.get('uploadImage')
        if uploaded_file:
            fs = FileSystemStorage()
            filename = fs.save(uploaded_file.name, uploaded_file)
            img_url = fs.url(filename)
            img_path = fs.path(filename)
            context['img'] = img_url

            # Load and preprocess the image
            try:
                img = cv2.imread(img_path, cv2.IMREAD_COLOR)
                if img is None:
                    raise ValueError("Error reading the image file.")

                img = cv2.resize(img, (224, 224))  # Resize for model
                img = img / 255.0  # Normalize image
                img = torch.tensor(img, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0).to(device)

                # Perform prediction
                with torch.no_grad():
                    outputs = model(img)
                    _, predicted = torch.max(outputs, 1)
                    prediction_index = predicted.item()
            except Exception as e:
                messages.error(request, f"Image processing or prediction failed: {e}")
                return render(request, 'profilepage.html', context)

            # Define diseases and diagnoses
            skin_disease_names = [
                'Cellulitis', 'Impetigo', 'Athlete Foot', 'Nail Fungus',
                'Ringworm', 'Cutaneous Larva Migrans', 'Chickenpox', 'Shingles'
            ]
            diagnosis = [
                'Cellulitis is a bacterial infection of the skin characterized by redness, swelling, and pain.',
                'Impetigo is a highly contagious bacterial infection that causes red sores on the skin.',
                'Athlete Foot is a fungal infection that causes itching, stinging, and burning between the toes.',
                'Nail Fungus is an infection of the nails caused by fungi, resulting in discoloration and thickening.',
                'Ringworm is a fungal infection that causes a circular rash with a red, scaly border.',
                'Cutaneous Larva Migrans is a skin infection caused by larvae of hookworms, leading to itchy, raised tracks on the skin.',
                'Chickenpox is a highly contagious viral infection that causes itchy rashes and blisters.',
                'Shingles is a viral infection that causes a painful, blistering rash, typically in a band on one side of the body.'
            ]

            # Get results
            result1 = skin_disease_names[prediction_index] if 0 <= prediction_index < len(skin_disease_names) else "Unknown Disease"
            result2 = diagnosis[prediction_index] if prediction_index < len(diagnosis) else "Diagnosis not available"
            context.update({'obj1': result1, 'obj2': result2})
        else:
            messages.error(request, "Please select an image to upload.")
    
    return render(request, 'profilepage.html', context)


# About page view
def about(request):
    return render(request, 'about.html')
