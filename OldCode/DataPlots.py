import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import skrf as rf
import pylab as py

# Finds data file given Case label, diameter, depth and position index
# Returns Network object or -1 if file was not found
def getData(xx,yy,mm,nn, domain = 's', path = "Data\\"):
    path = path + xx + "_Diameter" + yy + "_Depth" + mm + "_" + nn + ".s1p"
    try:
        data = rf.Network(path)
    except FileNotFoundError:
        data = -1
        print(f"Error Finding Data File, File that was checked: {path}")
    if data != -1:
        if domain == "impulse":
            data = data.impulse_response()
        elif domain == "step":
            data = data.step_response()

    return data

    
if __name__ == "__main__":
    data1 = getData("01","10","10","1")
    data2 = getData("01","10","10","2")
    data3 = getData("02","10","30","1")
    data4 = getData("03","10","50","1")
    print(data1)



    # Time Domain Plots - Impulse Response
    time1_impulse = getData("01","10","10","1", "impulse")
    time3_impulse = data3.impulse_response()
    time4_impulse = data4.impulse_response()
    py.figure(7)
    py.plot(time1_impulse[0][10:2085], time1_impulse[1][10:2085], 'g--')
    py.plot(time3_impulse[0][10:2085], time3_impulse[1][10:2085], 'b--')
    py.plot(time4_impulse[0][10:2085], time4_impulse[1][10:2085], 'r--')
    plt.xlim(-0.02 * 10**(-7),0.07 * 10**(-7))
    py.show()
