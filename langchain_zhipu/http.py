import requests
import json

def action_get(request: str):
    url = f'{DEFAULT_BASE_URL}/{request}'
    headers = {
        'Authorization': f'Bearer {generate_token()}',
        'Content-Type': 'application/json',
    }

    response = requests.get(url, headers=headers)
    
    return response

def action_post(request: str, data):
    url = f'{DEFAULT_BASE_URL}/{request}'
    headers = {
        'Authorization': f'Bearer {generate_token()}',
        'Content-Type': 'application/json',
    }

    response = requests.post(url, headers=headers, data=json.dumps(data))
    
    return response

def action_put(request: str, id: str, data):
    url = f'{DEFAULT_BASE_URL}/{request}/{id}'
    headers = {
        'Authorization': f'Bearer {generate_token()}',
        'Content-Type': 'application/json',
    }

    response = requests.put(url, headers=headers, data=json.dumps(data))
    
    return response
