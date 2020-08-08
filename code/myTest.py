
import os
import pandas as pd
import csv

# Path1 = 'home'
# Path2 = 'develop'
# Path3 = 'code'
# Path10 = Path1 + Path2 + Path3
# Path20 = os.path.join(Path1, Path2, Path3)
# print('Path10 = ', Path10)
# print('Path20 = ', Path20)
# 输出
#
# Path10 = homedevelopcode
# Path20 = home\develop\code

# 数据整理，统计故障数据数据个数

if __name__=='__main__':


    filenames = os.listdir(r'F:\QSLS')

    for data_row in filenames:
        # print(data_row)
        Path1 = 'F:\\'
        Path2 = 'QSLS'
        Path3 = data_row
        path = os.path.join(Path1, Path2, Path3)

        with open(path) as data:
            count = 0
            reader = csv.reader(data)
            next(reader)
            for data_row in reader:
                for i in data_row[84]:
                    if i == '1':
                        count=count+1;
            if count >2 :
                print("文件名为：%s,故障标签有：%d个,"%(path,count))




