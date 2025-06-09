# -*- coding: utf-8 -*-
"""
Created on Wed Oct 23 15:52:16 2024

@author: gmorena
"""

import pandas as pd
from API import IdeaConnectionClient
import math
import os
# _____________________________________________________________________________
# TO CHANGE:
# Choose Template:
    # 0: No Stiffner, No Widner
    # 1: Stiffner in column, No widener
    # 2: Stiffner in column + Widener
    # 3: Stiffner in column + in beam + Widener

PATH = r"C:\Users\benja\OneDrive\Desktop\Masterarbeit\Code\API_Worflow\Output_file_gen"

Template = 2
# Name of Input File
#InputFileName = 'Knee_widener_I_Output_red_Q4.parquet'
InputFileName = os.path.join(PATH,"Knee_widener_I_Output_red_Q4.parquet")
# Name of Output File
# OutputFileName = 'Knee_widener_I_Output_red_Q4.parquet'

#OutputFileName = 'Input_Knee_widener_I_Output_red_Q4.parquet'
OutputFileName = os.path.join(PATH,"Knee_widener_I_Output_red_Q4.parquet")


# Define which quadrant of the diagram we want
Quadrant = 'Q1'
# _____________________________________________________________________________


if Quadrant == 'Q1':
    a=1
elif Quadrant == 'Q4':
    a=-1
    
if Template == 0:
    IdeaName = 'Corner.ideaCon'
elif Template == 1:
    IdeaName = 'Corner_Stiffner.ideaCon'
elif Template == 2:
     IdeaName = r"C:\Users\benja\OneDrive\Desktop\Masterarbeit\Code\API_Worflow\Output_file_gen\Corner_Stiff_Wid.ideaCon"
     #IdeaName = r'Output_file_gen\Corner_Stiff_Wid.ideaCon'   
elif Template == 3:
    IdeaName = 'Corner_Stiff2_Wid.ideaCon'   
 
filename = IdeaName
conn = IdeaConnectionClient()
conn.open(filename)
inputs = pd.read_parquet(InputFileName)
outputs = inputs

# for i in range(22000,len(inputs)):
#for i in range(143000,144001):
for i in range(143000,143002):
    print('i =', i)
    params = {
        'Prof_c': inputs.at[i, 'Profile_x'], 'Prof_b': inputs.at[i, 'Profile_y'],
        'steel_grade': inputs.at[i, 'Steel grade_x'], 'gamma': inputs.at[i, 'Gamma'], 
        'h_wid': inputs.at[i, 'h_wid']/1000, 'b_wid': inputs.at[i, 'b_wid']/1000,
        'd_wid': inputs.at[i, 'd_wid']/1000, 't_fwid': inputs.at[i, 't_fwid']/1000,
        't_wwid': inputs.at[i, 't_wwid']/1000, 
        't_stiffc': inputs.at[i, 't_stiffc']/1000, 't_stiffb': inputs.at[i, 't_stiffb']/1000,
        'offset': inputs.at[i, 'Offset']/1000  
    }
        
    conn.update_params_by_name(params)
    if not conn.params_valid():
        raise Exception("Parameters validation failed.")
        
    # In case of overload, get in the loop and increase the load:
    def overloaded(calculated):
        k=0
        while calculated == False:
            print('k =', k)
            loads = conn.get_loads()
            if inputs.at[i,'M']==0:
                inputs.at[i,'V']*=0.9
            elif inputs.at[i,'V']==0:
                inputs.at[i,'M']*=0.9
            else:
                alpha = math.atan(inputs.at[i,'V']/inputs.at[i,'M'])
                r = (inputs.at[i,'M']**2+inputs.at[i,'V']**2)**0.5
                r*=0.9
                inputs.at[i,'M'] = r*math.cos(alpha)
                inputs.at[i,'V'] = r*math.sin(alpha)
            loads[0]['forcesOnSegments'][1]['my'] = int(inputs.at[i,'M']*1000)
            loads[0]['forcesOnSegments'][1]['qz'] = a*int(inputs.at[i,'V']*1000) #!!!!
            conn.set_loads(loads)
            print('M=',loads[0]['forcesOnSegments'][1]['my'] )
            print('V=',loads[0]['forcesOnSegments'][1]['qz'] ) 
            calculated = conn.calculate()
            k=(k+1)
            if k == 100:
                break
            
    # In case of underload,, get in the loop and decrease the load:
    def underloaded(applied_percentage):
        j=0
        while applied_percentage == 1:
            print('j =', j)
            print('Applied percentage =',applied_percentage)
            loads = conn.get_loads()
            if inputs.at[i,'M']==0:
                inputs.at[i,'V']*=1.1
            elif inputs.at[i,'V']==0:
                inputs.at[i,'M']*=1.1
            else:    
                alpha = math.atan(inputs.at[i,'V']/inputs.at[i,'M'])
                r = (inputs.at[i,'M']**2+inputs.at[i,'V']**2)**0.5
                r*=1.3
                inputs.at[i,'M'] = r*math.cos(alpha)
                inputs.at[i,'V'] = r*math.sin(alpha)
            loads[0]['forcesOnSegments'][1]['my'] = int(inputs.at[i,'M']*1000)
            loads[0]['forcesOnSegments'][1]['qz'] = a*int(inputs.at[i,'V']*1000) #!!!!
            conn.set_loads(loads)
            print('M=',loads[0]['forcesOnSegments'][1]['my'] )
            print('V=',loads[0]['forcesOnSegments'][1]['qz'] )    
            calculated = conn.calculate()
            if calculated == False:
                overloaded(calculated)
            
            # Get the applied loads:
            results = conn.get_results()
            loads = conn.get_loads()
            applied_percentage = results['analysis']['1']['appliedLoadPercentage']
            j=(j+1)
   
    loads = conn.get_loads()
    loads[0]['forcesOnSegments'][1]['my'] = int(inputs.at[i,'M']*1000)
    loads[0]['forcesOnSegments'][1]['qz'] = a*int(inputs.at[i,'V']*1000)#!!!!
    conn.set_loads(loads) 
    print('M=',loads[0]['forcesOnSegments'][1]['my'] )
    print('V=',loads[0]['forcesOnSegments'][1]['qz'] )            
    calculated = conn.calculate()
    if calculated == False:
        overloaded(calculated)
        
    # Get the applied loads:
    results = conn.get_results()
    loads = conn.get_loads()
    applied_percentage = results['analysis']['1']['appliedLoadPercentage']   
    if applied_percentage == 1:
        underloaded(applied_percentage)
    
    loads = conn.get_loads()
    loads[0]['forcesOnSegments'][1]['my'] = int(inputs.at[i,'M']*1000)
    loads[0]['forcesOnSegments'][1]['qz'] = a*int(inputs.at[i,'V']*1000)#!!!!
    conn.set_loads(loads)        
    calculated = conn.calculate()
    results = conn.get_results()
    loads = conn.get_loads()
    applied_percentage = results['analysis']['1']['appliedLoadPercentage']  
    applied_M = applied_percentage * loads[0]['forcesOnSegments'][1]['my']
    applied_V = applied_percentage * loads[0]['forcesOnSegments'][1]['qz']

    # Print the results in the output file:
    outputs.at[i, 'proz'] = applied_percentage
    outputs.at[i, 'M_Rd'] = applied_M/1000 # [kNm]
    outputs.at[i, 'V_Rd'] = applied_V/1000 # [kN]

    cost = results['costEstimationResults']['totalEstimatedCost']
    outputs.at[i, 'cost'] = cost
    
    # Every 500 the file is saved
    if i % 500 == 0:
        outputs.to_parquet(OutputFileName)

outputs.to_parquet(OutputFileName)


# myparquet = r'T:/02_Forschung und Entwicklung/202310_Madesco_IDEA-EUREKA/02_Simulation/GenerationOutput/Knee_widener_I_Output_red_Q4_testfile.parquet'
# df_3 = open_df_knee_joint_parq(myparquet)#, usecols="A:AS")

# excel_path = myparquet.replace('.parquet', '.xlsx')  # Replace extension
# df_3.to_excel(excel_path, index=False)

# mycsv = r'T:/02_Forschung und Entwicklung/202310_Madesco_IDEA-EUREKA/02_Simulation/GenerationOutput/Knee_widener_I_Output_red_Q4_testfile.csv'
# df_test = pd.read_csv(mycsv)
















