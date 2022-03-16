import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import skrf as rf
import pylab as py

# Finds data file given Case label, diameter, depth and position index
# Returns Network object or -1 if file was not found
def getData(xx,yy,mm,nn, path = "Data\\" ):
    path = path + xx + "_Diameter" + yy + "_Depth" + mm + "_" + nn + ".s1p"
    try:
        data = rf.Network(path)
    except FileNotFoundError:
        data = -1
        print(f"Error Finding Data File, File that was checked: {path}")
    return data

data1 = getData("01","10","10","1")
data2 = getData("01","10","10","2")
data3 = getData("02","10","30","1")
data4 = getData("03","10","50","1")
print(data1)

# Plot data from adjacent measurment positions
py.figure(1)
data1.plot_s_db()
data2.plot_s_db()

py.figure(2)
data1.plot_s_deg()
data2.plot_s_deg()

py.figure(3)
data1.plot_s_smith()
data2.plot_s_smith()
py.show()

# Plot Data for same position and diameter but different depths
py.figure(4)
data1.plot_s_db()
data3.plot_s_db()
data4.plot_s_db()
py.show()

# Check error handleing of getData function
data5 = getData("52", "10", "30", "37")
try:
    assert data5 == -1
except AssertionError:
    print("Error did not trigger")

