import json
with open('data/olive_diseases/train/_annotations.coco.json', 'r') as f:
    data = json.load(f)
    categories = sorted(data['categories'], key=lambda x: x['id'])
    print([c['name'] for c in categories])
