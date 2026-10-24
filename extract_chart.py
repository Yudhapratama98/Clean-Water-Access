import json
import base64

with open('Forecast-Daerah-Tertinggal-ByIsland.ipynb', 'r') as f:
    nb = json.load(f)

# Find the plot for an island with a single district (e.g. Bali & Nusa Tenggara has Alor, Jawa has Lebak)
# We will just extract the first output image that looks small or any image to show as an example
img_data = None
for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        for output in cell.get('outputs', []):
            if output.get('output_type') == 'display_data' and 'image/png' in output.get('data', {}):
                img_data = output['data']['image/png']
                # Try to pick the second or third image which might be a smaller grid (like Jawa)
                if len(img_data) < 200000: # Smaller file size typically means fewer plots
                    break
        if img_data and len(img_data) < 200000:
            break

if not img_data: # Fallback to first image
    for cell in nb['cells']:
        if cell['cell_type'] == 'code':
            for output in cell.get('outputs', []):
                if output.get('output_type') == 'display_data' and 'image/png' in output.get('data', {}):
                    img_data = output['data']['image/png']
                    break
        if img_data: break

if img_data:
    with open('forecast_q1_island.png', 'wb') as f:
        f.write(base64.b64decode(img_data))
    print("Chart saved")
