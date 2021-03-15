import cv2

def draw_box_save(boxes, labels, scores, name, file):
    # labels = [3, 3, 3, 1, 1, 3]
    # boxes = [[  0.0000, 140.9830,  32.2856, 158.6309],
    #         [ 97.5602, 139.4682, 109.9721, 147.9388],
    #         [ 96.1670, 133.1516, 111.1255, 148.7347],
    #         [  0.0000,  72.1410, 243.4181, 176.4464],
    #         [  0.0000,  47.1460, 253.9212, 251.0451],
    #         [144.4405, 129.6052, 159.9256, 144.8925]]

    COCO_INSTANCE_CATEGORY_NAMES = [
        '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
        'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
        'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
        'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
        'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
        'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
        'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
        'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
        'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
        'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
        'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
        'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
    ]

    
    with open(f'output_imgs/nicer/file_{name}.txt','a') as f:
        for label in labels:
            print(COCO_INSTANCE_CATEGORY_NAMES[label])
            f.write(COCO_INSTANCE_CATEGORY_NAMES[label]+"\n")

    # path = fr'file.png'
    # image = cv2.imread(path)
    image = cv2.imread(file)
    cv2.imwrite(f'output_imgs/nicer/file_{name}.png',image)
    # image = file
    for idx,box in enumerate(boxes):
        start=(int(boxes[idx][0]), int(boxes[idx][1]))
        end=(int(boxes[idx][2]), int(boxes[idx][3]))
        color=(0,0,255)
        thick = 1
        per = int(scores[idx]*100)
        if per>50:
            img = cv2.rectangle(image, start, end, color, thick)
            lab = COCO_INSTANCE_CATEGORY_NAMES[labels[idx]]
            img = cv2.putText(image,f"{per}%",start, cv2.FONT_HERSHEY_SIMPLEX, 0.3, thickness=1, color=(255,0,0))
            img = cv2.putText(image,f"{lab}",end, cv2.FONT_HERSHEY_SIMPLEX, 0.3, thickness=1, color=(255,0,0))

    cv2.imwrite(f'output_imgs/nicer/file_{name}_box.png',cv2.cvtColor(image, cv2.COLOR_BGR2RGB))