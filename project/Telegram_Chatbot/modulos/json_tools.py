import json
import sys

def convert_bool(obj):

    """
    --Input             dict
    --Output            dict

    Converts all booleans of a dict input intro strings. Necessary for analyze_cough.
    """
    if isinstance(obj, bool):
        return str(obj).lower()
    if isinstance(obj, (list, tuple)):
        return [convert_bool(item) for item in obj]
    if isinstance(obj, dict):
        return {convert_bool(key):convert_bool(value) for key, value in obj.items()}
    return obj