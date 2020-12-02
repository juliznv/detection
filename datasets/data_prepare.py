import os
import argparse

def get_arg():
    parser = argparse.ArgumentParser()
    parser.add_argument('-p','--path')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = get_arg()
    ann_dir = args.path
    anns = os.listdir(ann_dir)
    if not os.path.exists('./list'):
        os.mkdir('./list')
    if not os.path.exists('./annotations'):
        os.mkdir('./annotations')
    for d in anns:
        wf = open('list/' + d + '.txt', 'w+')
        fs = os.listdir(os.path.join(ann_dir, d))
        for f in fs:
            wf.write(f + '\n')
        wf.close()
    for d in anns:
        if not os.path.isdir(f'{ann_dir}/{d}'):
            continue
        os.system(f'python voc2coco.py ./list/{d}.txt {ann_dir}/{d} ./annotations/{d}.json')
        print(d, 'done.')