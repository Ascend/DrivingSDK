python3 -c "
file_path = '/usr/local/lib/python3.8/site-packages/mmcv_full-1.7.2-py3.8-linux-aarch64.egg/mmcv/parallel/_functions.py'
with open(file_path, 'r') as f:
    lines = f.readlines()
lines[74] = '            streams = [safe_get_stream(device) for device in target_gpus]\n'

function_code = [
        'def safe_get_stream(device):\\n',
        '    if isinstance(device, int):\\n',
        '        device = torch.device(f\'cuda:{device}\')\\n',
        '    return _get_stream(device)\\n'
    ]
lines[66:66] = function_code
with open(file_path, 'w') as f:
    f.writelines(lines)
"
