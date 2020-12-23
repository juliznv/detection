# VOC to COCO format
import os
import json
import argparse
import xml.etree.ElementTree as ET
from tqdm import tqdm

def get_args():
    parser = argparse.ArgumentParser(
        description = "Convert VOC to COCO format."
    )
    parser.add_argument('-d', '--data-path',type = str,
        default = './', help = 'path to dataset')
    parser.add_argument('-s', '--set_name', type = str,
        default = 'val', help = 'set name of dataset')
    return parser.parse_args()

def get(root, name, length):
    vars = root.findall(name)
    if len(vars) == 0:
        raise ValueError("Can not find %s in %s." % (name, root.tag))
    if length > 0 and len(vars) != length:
        raise ValueError(
            "The size of %s is supposed to be %d, but is %d."
            % (name, length, len(vars))
        )
    if length == 1:
        vars = vars[0]
    return vars

def convert(pdir, set_name):
    set_file = os.path.join(pdir, 'ImageSets', set_name + '.txt')
    xml_set = open(set_file, 'r').readlines()
    xml_files = [os.path.join(pdir, 'Annotations', xml[:-1] + '.xml') for xml in xml_set]
    bnd_id = 1
    categories = {}
    json_dict = {"images": [], "type": "instances", "annotations": [], "categories": []}
    for i, xml_file in tqdm(enumerate(xml_files)):
        tree = ET.parse(xml_file)
        root = tree.getroot()
        path = root.findall('path')
        if len(path) == 1:
            filename = os.path.basename(path[0].text)
        elif len(path) == 0:
            filename = get(root, "filename", 1).text
        else:
            raise ValueError("%d paths found in %s" % (len(path), xml_file))
        size = get(root, 'size', 1)
        width = get(size, 'width', 1).text
        height = get(size, 'height', 1).text
        image = {
            "file_name": filename,
            "height": height,
            "width": width,
            "id": i,
        }
        json_dict["images"].append(image)
        for obj in root.findall('object'):
            category = get(obj, "name", 1).text
            if category not in categories:
                categories[category] = len(categories)
            category_id = categories[category]
            bbox = get(obj, "bndbox", 1)
            xmin = int(get(bbox, "xmin", 1).text)
            ymin = int(get(bbox, "ymin", 1).text)
            xmax = int(get(bbox, "xmax", 1).text)
            ymax = int(get(bbox, "ymax", 1).text)
            ow = xmax - xmin
            oh = ymax - ymin
            assert xmax > xmin and ymax > ymin
            ann = {
                "area": ow * oh,
                "iscrowd": 0,
                "image_id": i,
                "bbox": [xmin, ymin, ow, oh],
                "category_id": category_id,
                "id": bnd_id,
                "ignore": 0,
                "segmentation": [],
            }
            json_dict["annotations"].append(ann)
            bnd_id += 1
    
    for cate, cid in categories.items():
        cat = {"supercategory": "none", "id": cid, "name": cate}
        json_dict["categories"].append(cat)

    json_file = os.path.join(pdir, 'Annotations', f'instances_{set_name}.json')
    
    json_fp = open(json_file, 'w')
    json.dump(json_dict, json_fp, indent = 1)
    json_fp.close()

if __name__ == '__main__':
    args = get_args()
    convert(args.data_path, args.set_name)
    
