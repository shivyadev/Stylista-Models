from fastapi import FastAPI, UploadFile, File
from torchvision import models, transforms
from contextlib import asynccontextmanager
from PIL import Image
import torch
import io

app = FastAPI()

model = None
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

@asynccontextmanager
async def lifespan(app: FastAPI):
    global model
    print("Loading EfficientNetB0 model...")
    model = models.efficientnet_b0(weights='IMAGENET1K_V1')
    model.fc = torch.nn.Identity()
    model.to(device)
    model.eval()
    print("Model loaded.")
    yield
    print("Shutting down...")

app = FastAPI(lifespan=lifespan)

@app.post("/embed")
async def get_image_embedding(image: UploadFile = File(...)):
    try:
        # Read the file content
        image_bytes = await image.read()
        
        # Debug: Check if we received any data
        if not image_bytes:
            raise ValueError("No image data received")
                    
        # Convert image bytes to PIL Image
        try:
            img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        except Exception as e:
            print(f"Error opening image: {str(e)}")
            raise ValueError(f"Invalid image format: {str(e)}")
        
        # Apply transformations
        input_tensor = transform(img).unsqueeze(0).to(device)

        with torch.no_grad():
            embeddings = model(input_tensor).squeeze().cpu().numpy().flatten()
        return {"embeddings": embeddings.tolist()}
    except Exception as e:
        print(f"Error processing image: {str(e)}")
        raise