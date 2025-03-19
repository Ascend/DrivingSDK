python3 -c "
file_path = '/usr/local/lib/python3.8/site-packages/mmcv_full-1.7.2-py3.8-linux-aarch64.egg/mmcv/parallel/_functions.py'
with open(file_path, 'r') as f:
    lines = f.readlines()
lines[78] = '            streams = [_get_stream(device) for device in target_gpus]\n'
del lines[66:70]
with open(file_path, 'w') as f:
    f.writelines(lines)
"

