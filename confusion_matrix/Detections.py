import pandas as pd
from yolo_net import Yolo
import argparse

parser = argparse.ArgumentParser(description='Detections')

parser.add_argument("-m", "--main", required=True,
help="main path is the directory which has the image list,coord_list,class_list,weight and cfg file ")

parser.add_argument("-i", "--img_list_name", required=True,
	help="image's list name (This list must have only images path) ")

parser.add_argument("-w", "--weight_name", required=True,
	help="weight's name ")

parser.add_argument("-c", "--cfg_name", required=True,
	help="config's name ")


args = vars(parser.parse_args())


file_df = pd.read_csv("{}/{}".format(args['main'],args['img_list_name']),header=None)
cnn = Yolo()
detections = pd.DataFrame()
cord_pred  = []
class_pred = []


for i in range(len(file_df)):
    image_path = file_df.iloc[i, 0]
    det = cnn.net(image_path,args['main'],args['weight_name'],args['cfg_name'])
    class_pred += [det['class_Ids']]
    cord_pred  += [det['boxes']]

    if len(det['class_Ids']) == 0:
        data_frame = pd.DataFrame(data=[[str(i),'None','None','None']])
        detections = detections.append((data_frame))
    else:
        for j in range(len(det['class_Ids'])):
            data_frame = pd.DataFrame(data=[[str(i),str(det['class_Ids'][j]),
                                      str(det['boxes'][j]),
                                      str(det['confidences'][j])]])
            detections = detections.append((data_frame))

with open('{}/cord_list.txt'.format(args['main']), 'w') as f:
    for item in cord_pred:
        f.write("%s\n" % item)

with open('{}/class_list.txt'.format(args['main']), 'w') as f:
    for item in class_pred:
        f.write("%s\n" % item)

detections.to_csv('{}/detections.txt'.format(args['main']),sep=' ',
                  index=False,header=['images','class_Ids','boxes',
                  'confidences'])