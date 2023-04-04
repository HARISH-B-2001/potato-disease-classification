from inspect import getmodule
from fastapi import FastAPI, File, UploadFile,Form
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf
from fastapi import FastAPI, Request, Form
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles



app = FastAPI()

app.mount("/static", StaticFiles(directory="disease/templates/static"), name="static")

templates = Jinja2Templates(directory="disease/templates")

origins = [
    "http://localhost",
    "http://localhost:3000",
]
app.add_middleware(CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def getmodel(name):
    MODEL= tf.keras.models.load_model("saved_models/"+name)
    return  MODEL
CLASS_NAMES_POTATO = ["Potato Early Blight", "Potato Late Blight", "Potato Healthy"]
CLASS_NAMES_BELL = ["Bell Pepper Bacterial spot","Bell Pepper Healthy"]
CLASS_NAMES_cherry = ["Cherry Powdery mildew", "Cherry healthy"]

@app.get("/")
async def home(request: Request):
    context = {"request": request}
    return templates.TemplateResponse("home.html", context)

@app.get("/ping")
async def ping():
    return "Hello, I am alive"

def read_file_as_image(data) -> np.ndarray:
    image = np.array(Image.open(BytesIO(data)))
    return image

@app.post("/predict")
async def predict(request: Request, name: str = Form(...), file: UploadFile = File(...)):

    if (name=="potato"):
        image = read_file_as_image(await file.read())
        img_batch = np.expand_dims(image, 0)
        MODEL=getmodel("potato")
        predictions = MODEL.predict(img_batch)
        predicted_class = CLASS_NAMES_POTATO[np.argmax(predictions[0])]
        confidence = np.max(predictions[0])
        context = {"request": request, "predicted_class": predicted_class, "confidence": confidence*100}
        return templates.TemplateResponse("predict.html", context)
    elif (name=="bell_pepper"):
        image = read_file_as_image(await file.read())
        img_batch = np.expand_dims(image, 0)
        MODEL=getmodel("bell_pepper")
        predictions = MODEL.predict(img_batch)
        predicted_class = CLASS_NAMES_BELL[np.argmax(predictions[0])]
        confidence = np.max(predictions[0])
        context = {"request": request, "predicted_class": predicted_class, "confidence": confidence*100}
        return templates.TemplateResponse("predict.html", context)
    elif (name=="cherry"):
        image = read_file_as_image(await file.read())
        img_batch = np.expand_dims(image, 0)
        MODEL=getmodel("cherry")
        predictions = MODEL.predict(img_batch)
        predicted_class = CLASS_NAMES_cherry[np.argmax(predictions[0])]
        confidence = np.max(predictions[0])
        context = {"request": request, "predicted_class": predicted_class, "confidence": confidence*100}
        return templates.TemplateResponse("predict.html", context)


if __name__ == "__main__":
    uvicorn.run(app, host='localhost', port=8000)

