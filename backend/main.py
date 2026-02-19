# Módulos
from Controller.routes import *
from Model.model import *

# System
import os
import io
from PIL import Image

# FastAPI
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware

import uvicorn

# Torch
import torch
import torch.nn.functional as F
from torchvision import models

# DotEnv
from dotenv import load_dotenv


# Loading .env
if load_dotenv():
    print(".env loaded successfully")
else:
    raise FileNotFoundError("couldn't load .env, verify if it is correct..")

# Application
app = FastAPI()

origins = [
    "*",
    # "http://localhost:3000",
    # "http://127.0.0.1:3000",
    # "http://localhost:5173",
    # "http://127.0.0.1:5173",
]

print("Origens: ", origins)

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)

# Loading Model
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TRANSFORM = Get_Transform()


# DogXCat Model
if os.path.isfile(os.getenv("MODEL_PATH")):
    LOADED_PET = torch.load(
        os.getenv("MODEL_PATH"), 
        map_location=DEVICE
    )
    print("DogXCat Model loaded successfully")
else:
    raise FileNotFoundError("couldn't identify the trained PET model archive")


# Dog && Cat Models
if os.path.isfile(os.getenv("DOG_MODEL_PATH")) and os.path.isfile(os.getenv("CAT_MODEL_PATH")):
    LOADED_DOG = torch.load(
        os.getenv("DOG_MODEL_PATH"), 
        map_location=DEVICE
    )
    print("Dog Model loaded successfully")

    LOADED_CAT = torch.load(
        os.getenv("CAT_MODEL_PATH"), 
        map_location=DEVICE
    )
    print("Cat Model loaded successfully")
else:
    raise FileNotFoundError("couldn't identify the trained Dog or Cat model archive")



# Setting DogXCat Model
NET = models.resnet18(weights=None)
num_features = NET.fc.in_features
NET.fc = nn.Linear(num_features, LOADED_PET["num_classes"])
NET.load_state_dict(LOADED_PET["model_state_dict"])
NET.eval().to(DEVICE)

PET_CLASSES = LOADED_PET["class_names"]


# Setting Dog Model
DOG = models.resnet18(weights=None)
num_features = DOG.fc.in_features
DOG.fc = nn.Linear(num_features, LOADED_DOG["num_classes"])
DOG.load_state_dict(LOADED_DOG["model_state_dict"])
DOG.eval().to(DEVICE)

DOG_CLASSES = LOADED_DOG["class_names"]


# Setting Cat Model
CAT = models.resnet18(weights=None)
num_features = CAT.fc.in_features
CAT.fc = nn.Linear(num_features, LOADED_CAT["num_classes"])
CAT.load_state_dict(LOADED_CAT["model_state_dict"])
CAT.eval().to(DEVICE)

CAT_CLASSES = LOADED_CAT["class_names"]


# Defs
def __preprocess_image(image_file: UploadFile):
    try:
        # Tratamento Imagem
        image_data = image_file.file.read()    
        image = Image.open(io.BytesIO(image_data)).convert("RGB")

        image_tensor = TRANSFORM(image).unsqueeze(0)
        return image_tensor

    except Exception as e:
        print(f"Error trying to manipulate the image: {e}")


def __predict_pet(image_tensor: torch.Tensor):

    try:
        with torch.no_grad():
            outputs = NET(image_tensor.to(DEVICE))
            probabilities = F.softmax(outputs, dim=1)
            predicted_class_idx = torch.argmax(probabilities, dim=1).item()
            confidence = probabilities[0][predicted_class_idx].item()
        
        predicted_class = PET_CLASSES[predicted_class_idx]

        all_probabilities = {
            PET_CLASSES[i]: float(probabilities[0][i])
            for i in range(len(PET_CLASSES))
        }

        return predicted_class, round(confidence * 100, 2), all_probabilities

    except Exception as e:
        print(f"Error trying to manipulate the image: {e}")


def __predict_dog(image_tensor: torch.Tensor):

    try:
        with torch.no_grad():
            outputs = DOG(image_tensor.to(DEVICE))
            probabilities = F.softmax(outputs, dim=1)
            predicted_class_idx = torch.argmax(probabilities, dim=1).item()
            confidence = probabilities[0][predicted_class_idx].item()
        
        predicted_class = DOG_CLASSES[predicted_class_idx]

        all_probabilities = {
            DOG_CLASSES[i]: float(probabilities[0][i])
            for i in range(len(DOG_CLASSES))
        }

        return predicted_class, round(confidence * 100, 2), all_probabilities

    except Exception as e:
        print(f"Error trying to manipulate the image: {e}")


def __predict_cat(image_tensor: torch.Tensor):

    try:
        with torch.no_grad():
            outputs = CAT(image_tensor.to(DEVICE))
            probabilities = F.softmax(outputs, dim=1)
            predicted_class_idx = torch.argmax(probabilities, dim=1).item()
            confidence = probabilities[0][predicted_class_idx].item()
        
        predicted_class = CAT_CLASSES[predicted_class_idx]

        all_probabilities = {
            CAT_CLASSES[i]: float(probabilities[0][i])
            for i in range(len(CAT_CLASSES))
        }

        return predicted_class, round(confidence * 100, 2), all_probabilities

    except Exception as e:
        print(f"Error trying to manipulate the image: {e}")


@app.get("/predict/")
def Debug():
    return {
        "class_names": LOADED_PET["class_names"],
        "num_classes": LOADED_PET["num_classes"],
        "model_info": LOADED_PET["model_info"],
        "num_dog_breeds": LOADED_DOG["num_classes"],
        "num_cat_breeds": LOADED_CAT["num_classes"],
    }

@app.post("/predict/")
async def predict( file: UploadFile = File(...) ):

    if not file.content_type.startswith('image/'):
        raise HTTPException(400, "Apenas imagens são aceitas")
    
    contents = await file.read()
    if len(contents) > 10 * 1024 * 1024:
        raise HTTPException(400, "Arquivo muito grande")
    
    await file.seek(0)

    image_tensor = __preprocess_image(image_file=file)

    # Predição
    predicted_pet, pet_confidence, all_probabilities = __predict_pet(image_tensor)
    match(predicted_pet):
        case "Dog":
            predicted_breed, breed_confidence, breed_probabilities = __predict_dog(image_tensor)
        case "Cat":
            predicted_breed, breed_confidence, breed_probabilities = __predict_cat(image_tensor)

    return {
        "filename": file.filename,
        "predicted_pet": predicted_pet,
        "pet": {
            "pet_confidence": pet_confidence,
            "all_probabilities": all_probabilities
        },
        "breed": {
            "predicted_breed": predicted_breed,
            "breed_confidence": breed_confidence,
            "breed_probabilities": breed_probabilities,
        }

    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)