import datetime

def time_format_convert(date, to_julian=True):
    if type(date) != 'str':
        date = str(date)

    leap_year = False
    year = int(date[:4])
    if (year % 4) == 0:
        if (year % 100) == 0:
            if (year % 400) == 0:
                leap_year = True
        else:
            leap_year = True

    if leap_year == False:
        md = [0, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31] #2017
    else:
        md = [0, 31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]

    if to_julian == True:
        month = int(date[4:6])
        day = int(date[6:8])

        julian_day = day
        for i in range(month):
            julian_day += md[i]
        julian_day = '00' + str(julian_day)
        julian_day = julian_day[-3:]

        return (str(year)+julian_day, int(year), int(julian_day))
    else:
        julian_day = date[4:7]
        month = 0
        day = int(julian_day)
        for i in md:
            if day > i:
                month = month + 1
                day = day - i
            else:
                break

        hour = 0
        minute = 0
        if len(date) >= 9:
            hour = int(date[7:9])
        if len(date) >= 11:
            minute = int(date[9:11])
        
        return datetime.datetime(year, month, day, hour, minute)


# 输入入datetime
def convert_datetime_to_julian(date):
    t = date.strftime("%Y%j%H%M")
    # return (year, jd, hour, minute, total)
    return (t[:4], t[4:7], t[7:9], t[9:11], t)


# 输入20172521622 hour和minute可以不存在
def convert_julian_to_datetime(julian_date):
    return time_format_convert(julian_date, to_julian=False)


if __name__ == "__main__":
    today = datetime.datetime.today()
    jd = convert_datetime_to_julian(today)
    print(type(jd))