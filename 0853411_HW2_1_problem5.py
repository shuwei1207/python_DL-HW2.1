# -*- coding: utf-8 -*-
"""
Created on Thu May 21 11:41:43 2020

@author: SeasonTaiInOTA
"""
import csv
import pygal.maps.world
from pygal_maps_world.i18n import COUNTRIES

def get_country_code(country_name):
    for code,name in COUNTRIES.items():
        if country_name == name:
            return  code
    return None 


filename="output.csv"
with open(filename) as f:   
    reader=csv.reader(f)   
    header_row=next(reader) 
    yes = {}
    no = {}          
    for row in reader:      
        if row[1]=='1':
            country = row[0]  
            code = get_country_code(country)
            a = int(row[1])
            if code:
                yes[code] = a
        else:
            country = row[0]
            code = get_country_code(country)
            b = int(row[1])
            if code:
                no[code] = b


wm=pygal.maps.world.World()
wm.title="武漢肺炎隔天增加數是否高於今天增加數"
wm.add("是",yes)
wm.add("否",no)
wm.render_to_file('world.svg')