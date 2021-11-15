import os
import cv2
import json
import operator
txtpath = 'yolov5/runs/detect/exp2/labels/'
img_path = 'dataset/images/test/'
entries = sorted(os.listdir(img_path), key=lambda x: int(os.path.splitext(x)[0]))
data = []

for i in range(len(entries)):
    img_name = entries[i].split(".")[0]
    if not os.path.isfile(txtpath+entries[i].replace('.png','.txt')):
        a = {"image_id": img_name, "bbox": (1,1,1,1), "score": 0.5, "label": 0}
    else:
        f = open(txtpath+entries[i].replace('.png','.txt'),'r')
        contents = f.readlines()

        im = cv2.imread('dataset/images/test/'+img_name + '.png')
        h, w, c = im.shape
        
        for content in contents:
            a = dict.fromkeys(['image_id', 'bbox', 'score', 'category_id', ])
            content = content.replace('\n','')
            c = content.split(' ')
            a['image_id'] = int(img_name)
            w_center = w*float(c[1])
            h_center = h*float(c[2])
            width = w*float(c[3])
            height = h*float(c[4])
            left = float(w_center - width/2)
            top = float(h_center - height/2)
            a['bbox'] = (tuple((left, top, width, height)))
            a['score'] = (float(c[5]))
            a['category_id'] = (int(c[0]))
            data.append(a)
        f.close()
json_object = json.dumps(data, indent=4)

print(len(data))
with open('answer.json', 'w') as fp:
    fp.write(json_object)