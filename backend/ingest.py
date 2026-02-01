import requests
import json

print('=' * 60)
print('Ingesting Knowledge Base...')
print('=' * 60)

try:
    response = requests.post('http://127.0.0.1:8000/ingest', timeout=60)
    if response.status_code == 200:
        result = response.json()
        status = result['status']
        chunks = result['chunks']
        print(f'✓ SUCCESS!')
        print(f'  Status: {status}')
        print(f'  Chunks created: {chunks}')
        print()
        print('The knowledge base is ready!')
        print('Go to http://localhost:5173 and start chatting!')
    else:
        print(f'✗ Error: {response.status_code}')
        print(response.text)
except Exception as e:
    print(f'✗ Error: {e}')
