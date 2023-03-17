import os
from typing import *
import py_vncorenlp

from pydantic import BaseModel
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import json

from multiprocessing import Process, Lock
mutex = Lock()

# py_vncorenlp.download_model(save_dir='vncorenlp')
model = py_vncorenlp.VnCoreNLP(save_dir='/home/data2/cuongnguyen/vncorenlp') # absolute path

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class Inputs(BaseModel):
    text: str = ""

@app.post("/vncorenlp")
def test_api(item: Inputs):
    text = item.text

    try:
      with mutex:
        output = model.annotate_text(text)
        res = {"sentences" :[
               [
                   {
                    "index": word['index'],
                    "form": word['wordForm'],
                    'posTag': word['posTag'],
                    'nerLabel': word['nerLabel'],
                    'head': word['head'],
                    'depLabel': word['depLabel']
                   }
                   for word in output[sentence]
               ] for sentence in output
            ]}
    except Exception as e: 
        print(e)
        try:
            with mutex:
                output = model.annotate_text(text)
                res = {"sentences" :[
                  [
                   {
                    "index": word['index'],
                    "form": word['wordForm'],
                    'posTag': word['posTag'],
                    'nerLabel': word['nerLabel'],
                    'head': word['head'],
                    'depLabel': word['depLabel']
                   }
                   for word in output[sentence]
                  ] for sentence in output
                ]}
        except Exception as e: 
            return e


    return res
