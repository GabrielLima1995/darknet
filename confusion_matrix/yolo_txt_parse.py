import pandas as pd

detections = pd.read_csv("cfg/result.txt",lineterminator="\n",sep=":",header=None)
detections.columns =['images','class_Ids','boxes','confidences']
names = pd.read_csv("cfg/buracos.names",sep="\n",header=None).values
detections['class_Ids'] = detections['class_Ids'].str.strip()

for i in range(len(names)):
    classes = str(names[i]).strip("[' ").strip(" ']")
    detections['class_Ids'].replace(classes, i, inplace=True)

class_ids = {}
for i in detections['images'].unique():
    class_ids[i] = [detections['class_Ids'][j] for j in detections[detections['images']==i].index]

for key in class_ids:
    for j in class_ids[key]:
        if isinstance(j, str):
            class_ids[key]=list([])

list_class_ids = class_ids.values()


cord_boxes = {}
for i in detections['images'].unique():
    cord_boxes[i] = [detections['boxes'][j] for j in detections[detections['images']==i].index]

list_cord_boxes=[]
for key in cord_boxes:
    det=[]
    for w in cord_boxes[key]:
       det += eval('[' + w + ']')
    list_cord_boxes+=[det]

with open('cfg/class_pred_22000_yolo_list.txt', 'w') as f:
     for item in list_class_ids:
         f.write("%s\n" % item)

with open('cfg/cord_pred_22000_yolo_list.txt', 'w') as f:
    for item in list_cord_boxes:
        f.write("%s\n" % item)

detections.to_csv('cfg/detections_yolo_22000.txt',sep=' ',index=False)