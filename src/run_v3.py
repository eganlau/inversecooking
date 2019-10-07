# Execute it like:
# http://localhost:3001/ingredients?filename=1.jpg

import torch
# import torch.nn as nn
# import numpy as np
import os
from args import get_parser
import pickle
from model import get_model
from torchvision import transforms
from utils.output_utils import prepare_output, get_ingrs
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import time
import pandas as pd
# import requests
# from io import BytesIO
import random
# from collections import Counter

from starlette.applications import Starlette
from starlette.responses import JSONResponse, HTMLResponse, RedirectResponse
import torch
from pathlib import Path
from io import BytesIO
import sys
import uvicorn
import asyncio

args = get_parser()
# args.maxseqlen = 15
args.ingrs_only=True
args.image_model='resnet152'

data_dir = '../data'

# code will run in gpu if available and if the flag is set to True, else it will run on cpu
use_gpu = True
device = torch.device('cuda' if torch.cuda.is_available() and use_gpu else 'cpu')
map_loc = None if torch.cuda.is_available() and use_gpu else 'cpu'


ingrs_vocab = pickle.load(open(os.path.join(data_dir, 'fittime_ingr_vocab.pkl'), 'rb'))
vocab = pickle.load(open(os.path.join(data_dir, 'fittime_instr_vocab.pkl'), 'rb'))
dataset = pickle.load(open(os.path.join(data_dir, 'train_dataset.pkl'), 'rb'))

ingr_vocab_size = len(ingrs_vocab)
instrs_vocab_size = len(vocab)

transf_list = []
transf_list.append(transforms.Resize(256))
transf_list.append(transforms.CenterCrop(224))
transform = transforms.Compose(transf_list)

t = time.time()
# args.embed_size=80
model = get_model(args, ingr_vocab_size, instrs_vocab_size)
# Load the trained model parameters
model_path = os.path.join("../saved_models/inversecooking/model/checkpoints", 'model.ckpt')
model.load_state_dict(torch.load(model_path, map_location=map_loc))
model.to(device)
model.eval()
model.ingrs_only = True
model.recipe_only = False
print ('loaded model')
print ("Elapsed time:", time.time() -t)

transf_list_batch = []
transf_list_batch.append(transforms.ToTensor())
transf_list_batch.append(transforms.Normalize((0.485, 0.456, 0.406), 
                                              (0.229, 0.224, 0.225)))
to_input_transf = transforms.Compose(transf_list_batch)

greedy = [True, False, False, False]
beam = [-1, -1, -1, -1]
temperature = 1.0
numgens = len(greedy)

show_anyways = False #if True, it will show the recipe even if it's not valid
image_folder = os.path.join(data_dir, 'demo_imgs/fittime')

demo_files = os.listdir(image_folder)

print("before app")
app = Starlette()

@app.route("/ingredients", methods=["GET"])
async def classify_url(request):

    image_path = os.path.join(image_folder, request.query_params["filename"])
    if not os.path.exists(image_path):
        return JSONResponse({"error": "file doesn't exist"})
    image = Image.open(image_path).convert('RGB')
    transf_list = []
    transf_list.append(transforms.Resize(256))
    transf_list.append(transforms.CenterCrop(224))
    transform = transforms.Compose(transf_list)
    
    image_transf = transform(image)
    image_tensor = to_input_transf(image_transf).unsqueeze(0).to(device)
    num_valid = 1
    result = []
    for i in range(numgens):
        with torch.no_grad():
            outputs = model.sample(image_tensor, greedy=greedy[i], 
                                   temperature=temperature, beam=beam[i], true_ingrs=None)
            
        ingr_ids = outputs['ingr_ids'].cpu().numpy()
        result.append({"ingredients_id":pd.Series(ingr_ids[0]).to_json(orient='values'),
                        "ingredients_name":get_ingrs(ingr_ids[0], ingrs_vocab)})
    print(result)

    return JSONResponse(result)
 

if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=3001)