import pandas as pd
from yolo_net import Yolo
import csv

config ='cfg/'
test_df = pd.read_csv(config+"test.txt",header=None)
cnn = Yolo()
detections = pd.DataFrame()
cord_pred  = []
class_pred = []


for i in range(len(test_df)):
    image_path = test_df.iloc[i, 0]
    det = cnn.net(image_path,config)
    class_pred += [det['class_Ids']]
    cord_pred  += [det['boxes']]

    if len(det['class_Ids']) == 0:
        data_frame = pd.DataFrame(data=[[str(i),'None','None','None']])
        detections = detections.append((data_frame))
    else:
        for j in range(len(det['class_Ids'])):
            data_frame = pd.DataFrame(data=[[str(i),str(det['class_Ids'][j]),str(det['boxes'][j]),str(det['confidences'][j])]])
            detections = detections.append((data_frame))

with open('cfg/cord_pred_22000_list.txt', 'w') as f:
    for item in cord_pred:
        f.write("%s\n" % item)

with open('cfg/class_pred_22000_list.txt', 'w') as f:
    for item in class_pred:
        f.write("%s\n" % item)

detections.to_csv(config+'detections_22000.txt',sep=' ',index=False,header=['images','class_Ids','boxes','confidences'])