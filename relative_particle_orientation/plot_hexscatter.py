'''
Author: Rasmus K Jensen
This script will take the output of the relative particle orientation script and create a hexbin plot
'''

import matplotlib.pyplot as plt


def create_plot(x,y,output_file):
    plt.hexbin(x,y,gridsize=(20),cmap='Greens') #,bins=None, edgecolors=None,)
    cb = plt.colorbar()
    plt.xlabel(r'$\Delta{}x$ (nm)')
    plt.ylabel(r'$\Delta{}y$ (nm)')
    ticks = [-250,-200,-150,-100,-50,0,0,50,100,150,200,250]
    str_ticks = [str(int(x/10)) for x in ticks]
    plt.xticks(ticks=ticks,labels=str_ticks)
    plt.yticks(ticks=ticks,labels=str_ticks)
#   plt.show()
    plt.savefig(output_file,dpi=1000,transparent=True)
    plt.close()

def read_data(filename):
    with open(filename, 'r') as f:
        data = f.readlines()
    x = [d.split(',')[0].strip() for d in data]
    y = [d.split(',')[1].strip() for d in data]
    return x,y

if __name__ == '__main__':
    for input,output in [('/g/scb/mahamid/rasmus/edmp/relative_15.csv','/g/scb/mahamid/rasmus/edmp/relative_15.csv.png')]:
        x,y = read_data(input)
        create_plot(x,y, output)