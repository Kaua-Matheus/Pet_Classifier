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

# Torch
import torch
import torch.nn.functional as F

# DotEnv
from dotenv import load_dotenv

# Application
app = FastAPI()

origins = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
)

# Loading .env
if load_dotenv():
    print(".env loaded successfully")
else:
    raise FileNotFoundError("couldn't load .env, verify if it is correct..")

# Loading Model
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TRANSFORM = Get_Transform()

# Loading Trained Model
if os.path.isfile(os.getenv("SAVED_MODEL_PATH")):
    LOADED_SAVE = torch.load(
        os.getenv("SAVED_MODEL_PATH"), 
        map_location=DEVICE
    )
    print("saved model loaded successfully")
else:
    raise FileNotFoundError("couldn't identify the trained model archive")

# Setting Model to Eval Mode
NET = NeuralNetwork(num_labels=LOADED_SAVE["num_classes"]).to(DEVICE)
NET.load_state_dict(LOADED_SAVE["model_state_dict"])
NET.eval()

CLASSES = LOADED_SAVE["class_names"]


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


@app.get("/predict/")
def Debug():
    return {
        "class_names": LOADED_SAVE["class_names"],
        "num_classes": LOADED_SAVE["num_classes"],
        "model_info": LOADED_SAVE["model_info"],
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
    with torch.no_grad():
        outputs = NET(image_tensor.to(DEVICE))
        probabilities = F.softmax(outputs, dim=1)
        predicted_class_idx = torch.argmax(probabilities, dim=1).item()
        confidence = probabilities[0][predicted_class_idx].item()
    
    predicted_class = CLASSES[predicted_class_idx]

    all_probabilities = {
        CLASSES[i]: float(probabilities[0][i])
        for i in range(len(CLASSES))
    }

    return {
        "filename": file.filename,
        "predicted_class": predicted_class,
        "confidence": round(confidence * 100, 2),
        "all_probabilities": all_probabilities,
    }