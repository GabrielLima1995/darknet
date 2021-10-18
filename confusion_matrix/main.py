import cv2
from ground_truth import glabels
from ast import literal_eval
import pandas as pd
import numpy  as np
from IoU import bb_intersection_over_union


class get():

    def matrix(self):

        confusion_matrix = np.zeros((12,12))
        conf='cfg/'
        ground_truth = glabels()
        bbox = ground_truth.truth()
        path = pd.read_csv(conf+"test.txt",header=None)
        #color   = (0, 0, 255)
        #colorp  = (0,255,  0)
        #fn = 0
        fp = 0
        tp = 0

        with open(conf+'cord_pred_22000_yolo_list.txt') as f:
            mainlist = [list(literal_eval(line)) for line in f]

        with open(conf+'class_pred_22000_yolo_list.txt') as f:
            classlist = [list(literal_eval(line)) for line in f]

        for i in range(len(path)):

            img = cv2.imread(path.iloc[i, 0])
            size = img.shape[:2]
            pred_bbox = mainlist[i]
            pred_class = classlist[i]
            classes_i    = []
            cordinates_i = []

            for k in bbox[i]:

                classes     = k[0]
                cordinates = k[1:]
                xc = cordinates[0]*size[1]
                yc = cordinates[1]*size[0]
                w  = int(cordinates[2]*size[1])
                h  = int(cordinates[3]*size[0])
                x  = int(xc - (w/2))
                y  = int(yc - (h/2))
                conv_cordinates =[x,y,x+w,y+h]
                #cv2.rectangle(img,(x,y),(x+w,y+h),color,1)
                classes_i    += [classes]
                cordinates_i += [conv_cordinates]

                correct_cord =[]
                for o in pred_bbox:
                    xp = o[0]
                    yp = o[1]
                    wp = o[2]
                    hp = o[3]
                    correct_cord += [[xp,yp,xp+wp,yp+hp]]
                    #cv2.rectangle(img,(xp,yp),(xp+wp,yp+hp),colorp,1)#cm


            #len_ground_truth = np.zeros(len(classes_i))

            #print("-------------")

            #print("image:",i)

            #iteration classes preditas

            for r in range(len(pred_class)):

                pred = correct_cord[r]
                #print("predito:",pred_class[r])
                best_iou = 0
                best_list =[-1,pred_class[r]]

                #iteration over ground truth
                for s in range(len(classes_i)):
                    #print("ground truth:",classes_i[s])
                    iou = bb_intersection_over_union(cordinates_i[s],pred)
                    #print(iou)

                    if iou > best_iou:
                        best_iou  = iou
                        best_list = [classes_i[s],pred_class[r]]
                        #print("iou > best")
                        #[ground_truth_class,predict_class]
                        #len_ground_truth[s] +=1

                #cv2.putText(img, 'iou' + str(best_iou), (pred[0] - 5, pred[1] - 5),
                           # cv2.FONT_HERSHEY_SIMPLEX, 0.6, colorp, 2)

                if best_iou > 0.5:

                    if best_list[0] == best_list[1]:

                        tp+=1
                        confusion_matrix[int(best_list[0]),best_list[1]] +=1
                        #print("best>0.5 tp ")

                    else:

                        fp +=1
                        confusion_matrix[int(best_list[0]),best_list[1]] +=1
                        #print("best>0.5 fp ")
                else:

                    fp += 1
                    confusion_matrix[-1,best_list[1]] += 1
                   # print("best<0.5 fp ")

            #print(confusion_matrix)

            #fn_array = np.where(len_ground_truth == 0)
            #fn += len(fn_array[0])

           # cv2.imshow("image",img)#cm
           # cv2.waitKey(0)#cm

        return confusion_matrix,fp,tp