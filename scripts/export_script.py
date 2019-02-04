import os

dataFolder = '/gruvi/Data/chenliu/ScanNet/scans/'
split = 'val'
with open('split_' + split + '.txt', 'r') as f:
    for line in f:
        scene_id = line.strip()
        if len(scene_id) != 12:
            continue
        filename = dataFolder + '/' + scene_id
        output_filename = 'instance_val/' + scene_id + '.txt'
        print('python scripts/export_train_mesh_for_evaluation.py --scan_path=' + filename + ' --output_file=' + output_filename)
        os.system('python scripts/export_train_mesh_for_evaluation.py --scan_path=' + filename + ' --output_file=' + output_filename)
        continue
    pass
