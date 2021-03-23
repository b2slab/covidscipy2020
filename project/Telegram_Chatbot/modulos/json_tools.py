import json
import sys

def convert_bool(obj):
    if isinstance(obj, bool):
        return str(obj).lower()
    if isinstance(obj, (list, tuple)):
        return [convert_bool(item) for item in obj]
    if isinstance(obj, dict):
        return {convert_bool(key):convert_bool(value) for key, value in obj.items()}
    return obj