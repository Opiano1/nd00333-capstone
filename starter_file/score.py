import json
import numpy as np
import os
from azureml.core.model import Model


def init():
    global model_path
    model_path = Model.get_model_path(model_name='BestModel')

def run(data):
    try:
        data = np.array(json.loads(data))
        result = model.predict(data)
        # You can return any data type, as long as it is JSON serializable.
        return result.tolist()
    except Exception as e:
        error = str(e)
        return error