from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision import transforms
from PIL import Image
import torch
import io

app = FastAPI()

# Load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = maskrcnn_resnet50_fpn(pretrained=True).to(device).eval()

@app.post("/detect")
async def detect_objects(image: UploadFile = File(...)):
    try:
        image_bytes = await image.read()
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        img_tensor = transforms.ToTensor()(img)
    
        with torch.no_grad():
            prediction = model([img_tensor])
        
        # Simplified response
        scores = prediction[0]['scores'].cpu().numpy()
        labels = prediction[0]['labels'].cpu().numpy()
        masks = prediction[0]['masks'].cpu().numpy()

        return JSONResponse(content={
            "masks": masks.tolist(),
            "labels": labels.tolist(),
            "scores": scores.tolist()
        })

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)
