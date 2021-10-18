import pandas as pd

class glabels():

    def truth(self):

        current_dir = "/home/Dados/datasets_images/image_pmsp/buracos_10"
        conf ='cfg/'
        df = pd.read_csv(conf+"test.txt",header=None)
        ground_truth = pd.DataFrame()
        bbox = []

        for i in range(len(df)):
            image_path = current_dir + df.iloc[i,0].split("buracos_10")[-1].split(".")[-2]+".txt"
            text_file = pd.read_csv(image_path, header=None, delimiter=" ")
            empty = []

            for j in range(len(text_file)):
                row_image = text_file.iloc[j,:]
                list      = row_image.values.tolist()
                empty     += [list]
                dict = {'image': i,'class_Ids':text_file.iloc[j,0],
                        'boxes': [text_file.iloc[j,1],text_file.iloc[j,2],text_file.iloc[j,3],text_file.iloc[j,4]]}
                pd_row = pd.DataFrame(data=[[dict['image'],dict['class_Ids'],dict['boxes']]])
                ground_truth = ground_truth.append(pd_row)
            bbox += [empty]
        ground_truth.to_csv(conf+'ground_truth.txt',sep=' ',index=False,header=['image','class_Ids','boxes'])
        return bbox