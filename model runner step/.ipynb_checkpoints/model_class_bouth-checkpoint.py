import mlrun
import numpy as np

from datetime import datetime
from mlrun.serving.states import Model, ModelSelector
from typing import Optional
from cloudpickle import load
from storey import MapClass
import time
import asyncio




class MyModel(Model):
    def __init__(self, *args, artifact_uri: str = None, raise_exception = False, **kwargs):
        super().__init__(*args, artifact_uri=artifact_uri, raise_exception=raise_exception, **kwargs)
    
    def load(self):
        """load and initialize the model and/or other elements"""
        model_file, _ = self.get_local_model_path(".pkl")
        self.model = load(open(model_file, "rb"))
    
    async def predict(self, body: dict, **kwargs) -> dict:
        """Generate model predictions from sample."""
        print("sync")
        
        # time.sleep(1)
        await asyncio.sleep(1)
        
        feats = np.asarray(body["inputs"]["here"])
        result: np.ndarray = self.model.predict(feats)
        return {"outputs": {"label" :result.tolist()}}

    async def predict_async(self, body: dict, **kwargs) -> dict:
        """Generate model predictions from sample."""
        
        await asyncio.sleep(1)
        
        print("async")
        feats = np.asarray(body["inputs"]["here"])
        result: np.ndarray = self.model.predict(feats)
        return {"outputs": {"label" :result.tolist()}}

    
class MyModelSelector(ModelSelector):
    def __init__(self, name:str):
        super().__init__()
        self.name = name
    
    """Selector allows the ModelRunnerStep to run some of models provided based on the event"""
    def select(self, event, available_models: list[Model]) -> Optional[list[str]]:
        return event.body.get("models")
    
    
class MyEnrichStep(MapClass):
    def do(self, event):
        event["timestamp"] = str(datetime.now().timestamp())
        return event


class MyPreprocessStep(MapClass):
    def do(self, event):
        inputs = event.pop("inputs")
        event["inputs"] = {"here": inputs}
        return event
        
