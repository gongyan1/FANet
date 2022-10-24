import os
import glob
import math
import random


# train_night:  1004 1128 714
# train_day: 1721 1089 1945
# Train: 7601 day:4755 night:2846
#        campus  Road    Downtown
# Day:   set 00, set 01, set 02      4755    62%
# Night: set 03, set 04, set 05      2846    38%
#          2725    2217    2659




# 0 2284 800 917 567
# 0 562 204 211 147
# 数据集划分
#
#              campus   Road    Downtown
# Train: day     1355    882       1579      3816
#        night   800    917         567      2284
#               2155     1799       2146     6100
#
##              campus   Road    Downtown
# val:   day       366    207        366      939
#        night   204      211        147      562
#               570       418        513      1501


# test_day : 648 406 401
# test_night: 175 444 178
# Test: 2252 day:1455 night:797
#        campus  Road    Downtown
# Day:   set 06, set 07, set 08      1455
# Night: set 09, set 10, set 11      797
#           823    850    579


### count number
root_path = '/home/ubuntu/dataset/kaist_pixel_level/preparing_data/val/images'
txt_list = os.listdir(root_path)
night = ['set03','set04','set05','set09','set10','set11']
day = ['set00','set01','set02','set06','set07','set08']
campus = ['set00','set03','set06','set09']
Road = ['set01','set04','set07','set10']
Downtown = ['set02','set05','set08','set11']
#
day_count = 0
night_count = 0
campus_count = 0
Road_count = 0
Downtown_count = 0
campus_count_night = 0
Road_count_night = 0
Downtown_count_night = 0
campus_count_day = 0
Road_count_day = 0
Downtown_count_day = 0



for item in txt_list:
    tag = item[:5]
    # print(tag)
    # item_new = item[:11]+'lwir_'+item[19:]

    # day/night
    if tag in night:
        night_count = night_count+1
        if tag in campus:
            campus_count_night = campus_count_night + 1
        elif tag in Road:
            Road_count_night = Road_count_night + 1
        elif tag in Downtown:
            Downtown_count_night = Downtown_count_night + 1
        else:
            print(tag)
            exit()
    elif tag in day:
        day_count = day_count+1
        if tag in campus:
            campus_count_day = campus_count_day + 1
        elif tag in Road:
            Road_count_day = Road_count_day + 1
        elif tag in Downtown:
            Downtown_count_day = Downtown_count_day + 1
        else:
            print(tag)
            exit()

    else:
        print(tag)
        exit()


    if tag in campus:
        campus_count = campus_count+1
    elif tag in Road:
        Road_count = Road_count+1
    elif tag in Downtown:
        Downtown_count=Downtown_count+1
    else:
        print(tag)
        exit()
print("day_count","night_count","campus_count","Road_count","Downtown_count")
print(day_count,night_count,campus_count,Road_count,Downtown_count)
print("campus_count_day","Road_count_day","Downtown_count_day")
print(campus_count_day,Road_count_day,Downtown_count_day)
print("campus_count_night","Road_count_night","Downtown_count_night")
print(campus_count_night,Road_count_night,Downtown_count_night)
