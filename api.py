import loguru
import uvicorn
import torch
from uuid import uuid4

from utilities.endpoint_configs import getEndpointConfigManager
from utilities.data_models import TTSGenerationResponse
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from whisperspeech.pipeline import Pipeline
two_sa_model = 'collabora/whisperspeech:s2a-q4-tiny-en+pl.model'

pipe = Pipeline(s2a_ref=two_sa_model)
# this is very slow right now since our inference code is not very optimized
# but even without this crucial optimization it is still better than real-time on an RTX 4090

logger = loguru.logger

manager = getEndpointConfigManager()

tts_config = manager.tts

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/data", StaticFiles(directory="data"), name="data")


@logger.catch()
@app.post(f"{manager.version}{tts_config.endpoint}")
def generate_tts(payload: TTSGenerationResponse):
    logger.info(f"Received request: {str(payload)}")
    text = payload.prompt

    try:
        filename = str(uuid4())
        pipe.generate_to_notebook(f"data/{filename}.wav", text)
        return JSONResponse(status_code=200, content={"filename": filename})

    except HTTPException as e:
        logger.exception(e)
        return JSONResponse(status_code=500, content={"error": str(e)})

if not torch.cuda.is_available():
    raise ValueError(
        "Currently configured to run on a GPU but no GPU was detected!  \n\n"
    )




if __name__ == "__main__":
    logger.debug(
        f"Running on http://{tts_config.host}:{tts_config.port}{manager.version}{tts_config.endpoint}"
    )
    uvicorn.run("api:app", host=tts_config.host, port=int(tts_config.port), reload=True)
