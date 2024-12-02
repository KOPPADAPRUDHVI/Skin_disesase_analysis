# Skin Disease Detection

This project leverages advanced image processing techniques and machine learning to provide an accessible, cost-effective solution for diagnosing skin diseases. Developed by **Team 9**, the system processes skin images to identify abnormalities, aiding in the detection of conditions like eczema and melanoma.

## Features

- **User Authentication**: 
  - Login and Signup functionalities for secure access.
- **Profile Page**:
  - Allows users to upload images for analysis.
- **Image Processing**:
  - Utilizes OpenCV for extracting features like color, texture, and shape.
- **Disease Detection**:
  - Employs the DINO model with Vision Transformer (ViT) architecture to classify skin diseases with 92% accuracy.
- **Dynamic Image Display**:
  - Uploaded images are stored using `FileSystemStorage` and displayed on the profile page with dynamically generated URLs.

## Application Pages

1. **Home Page**:
   - Entry point with options to login or signup.
2. **Login Page**:
   - For existing users to access their profile.
3. **Signup Page**:
   - Enables new users to create accounts.
4. **Profile Page**:
   - Allows users to upload skin images for disease detection.


## Technology Stack

- **Frontend**: HTML, CSS, JavaScript
- **Backend**: Django/Flask (assumed based on functionality)
- **Database**: SQLite- **Image Processing**: OpenCV
- **Machine Learning**: DINO model with Vision Transformer (ViT)

## How to Run

1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/SkinDiseaseDetection.git

2. Navigate to the project directory:

   cd SkinDiseaseDetection

3.Install the required dependencies:
   pip install -r requirements.txt

4. Run the server:

   python manage.py runserver

5. Open your browser and visit:

   http://127.0.0.1:8000/




Feel free to adjust the instructions or details based on specific technologies or configurations in your implementation.



