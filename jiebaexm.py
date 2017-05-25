# -*- coding: utf-8 -*-
"""
Created on Tue Nov 15 21:00:03 2016

@author: Ryan
"""
import jieba

read_place = "E:\\BIIC\\jdth\\"
output_place = "E:\\BIIC\\jdth_split\\"

jieba.set_dictionary('C:\\Users\\TACO\\Anaconda3\\Lib\\site-packages\\jieba\\dict.txt.big')
filename='E:\\BIIC\\jdth\\1.txt';
count = 0
while count < 300:
    count = count + 1;
    read_filename = read_place + str(count) + ".txt"
    output_filename = output_place + str(count) + ".txt"
    content = open(read_filename,'r+', encoding = 'utf-8').read()
    content= content.strip('\n')
    words = jieba.cut(content, cut_all=False)
    #==============================================================================
    print("Output 精確模式 Full Mode：")
    output = open(output_filename,'w+',encoding = 'utf-8')
    jieout=[]
    for word in words:
        jieout.append(word)
    print(jieout)
    for point in jieout:
        output.write(str(point))
        output.write("\n")
    time.sleep(10)
