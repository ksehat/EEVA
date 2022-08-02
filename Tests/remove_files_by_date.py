import os
import time

os.chdir('D:/Python projects/EEVA/trader/Optimization '
         'results/All combinations run/Sys1_2_2/IOTAUSDT 30m')
for f in os.listdir():
    if float([x for x in time.ctime(os.path.getmtime(f)).split(' ') if ':' in x][0].split(':')[
                 0]) <10:
        os.remove(f)
