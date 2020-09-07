import os
import shutil
import time
import subprocess

import sys
if __name__ == "__main__":
    sys.path.append(os.getcwd())   



localdir = './DATA/ABI-L1b-RadM-New/'
if os.path.exists(localdir) == False:
    os.makedirs(localdir)

cachedir = './DATA/ABI-L1b-RadM/'

sensorlist = ['ABI-L1b-RadM']
# sensorlist = ['ABI-L1b-RadC', 'ABI-L1b-RadF', 'ABI-L1b-RadM', 'ABI-L2-CMIPC', 
#               'ABI-L2-CMIPF', 'ABI-L2-CMIPM', 'ABI-L2-MCMIPC', 'ABI-L2-MCMIPF',
#               'ABI-L2-MCMIPM']

daylist = list(range(242, 276+1))
# irma_range = [242, 259] #Formed: August 30, 2017; Dissipated: September 16, 2017
# maria_range = [259, 276]  #Formed: September 16, 2017; Dissipated: October 3, 2017
# jose_range = [248, 269] #Formed: September 5, 2017; Dissipated: September 26, 2017

channellist = ['M3C01', 'M3C07', 'M3C09', 'M3C14', 'M3C15']
#channellist = ['C01','C02','C03','C04','C05','C06','C07','C08','C09','C10','C11','C12','C13','C14','C15','C16']

# ORI_ABI-L1b-RadM1-M3C01_G16_s20172420000247_exx_cxx.nc
def cut_file_name(f):
    seg = f.split('_')
    sensor = seg[1][:-6]
    channel = seg[1][-5:]
    date = seg[3][1:-3]  #!
    return (sensor, channel, date)


def classify_data(candidacy):
    channel_date_list = []
    for _ in channellist:
        channel_date_list.append({})

    for cand in candidacy:
        name = cand[1]
        sensor, channel, date = cut_file_name(name)
        for i in range(len(channellist)):
            if channel == channellist[i]:
                key = sensor+date
                channel_date_list[i][key] = cand
                break
    return channel_date_list
    

def pull_data(data_cache):
    for data in data_cache:
        size =  int(data[0])
        name = data[1]

        file_name = name.split('/')[-1]

        cache_file = os.path.join(cachedir, file_name)
        local_file = os.path.join(localdir, file_name)

        # if os.path.exists(local_file):
        #     print(os.path.getsize(local_file))

        if os.path.exists(local_file) == False or os.path.getsize(local_file) != size:

            if os.path.exists(cache_file) and os.path.getsize(cache_file) == size:
                shutil.copy( os.path.join(cachedir, file_name) , localdir)
                print("copy : %s" % (data))
                continue

            rccloc = 'aws s3 cp s3://noaa-goes16-hurricane-archive-2017/' + name + ' ' + localdir
            pullcmd = rccloc + ' --no-sign-request --endpoint-url https://griffin-objstore.opensciencedatacloud.org'
            #print(pullcmd)
            _ = subprocess.Popen(pullcmd, shell=True, stdout = subprocess.PIPE)
            print("Downloading : %s" % (data))
            time.sleep(0.5)

def download_data(channel_date_list, chosen_times = 5):
    main_channel = channel_date_list[0]
    main_channel_key = list(main_channel.keys())

    count = 0
    data_cache = []

    for mck in main_channel_key:
        if (count % chosen_times) != 0:
            count += 1
            continue 
        count += 1

        for channel in channel_date_list:
            if mck in channel:
                data_cache.append(channel[mck])
            else:
                data_cache.clear()
                break
        if len(data_cache) == len(channellist):
            pull_data(data_cache)
        data_cache.clear()
        


precmd = 'aws s3 --no-sign-request ls --endpoint-url https://griffin-objstore.opensciencedatacloud.org  --recursive noaa-goes16-hurricane-archive-2017/'

for sensor in sensorlist:
    for day in daylist:
        cmd = precmd + sensor + '/' + str(day) + '/' + ' | awk \'{print $3";"$4}\''
        #print(cmd)
        datalist = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE)
        result_lines = datalist.stdout.readlines()

        candidacy = []

        #ABI-L1b-RadM/242/00/ORI_ABI-L1b-RadM1-M3C01_G16_s20172420000247_exx_cxx.nc
        for result_line in result_lines:
            result_line = str(result_line, 'UTF-8')[:-1]
            candidacy.append(result_line.split(";"))
        #sorted(candidacy)

        camera = {}
        for cand in candidacy:
            name = candidacy[1]
            sensor, _, _ = cut_file_name(name)
            camera_number = sensor[-1]

            if camera_number in camera == False:
                camera[camera_number] = []
            camera[camera_number].append(cand)

        camera_key = list(camera.keys())
        for ck in camera_key:
            channel_date_list = classify_data(camera[ck])
            download_data(channel_date_list)