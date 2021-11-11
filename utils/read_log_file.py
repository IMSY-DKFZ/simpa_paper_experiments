from datetime import datetime


def read_log_file(path):
    with open(path) as file:
        f = file.readlines()

    return f


def get_info_fields_from_log_file(path):
    lines = read_log_file(path)
    info_fields = [line for line in lines if line.split(" ")[3] == "INFO"]
    first_line = lines[0]
    return [first_line] + info_fields


def get_times_from_log_file(path):
    info_fields = get_info_fields_from_log_file(path)

    first_line = info_fields[0]
    vol_creation_start = info_fields[1]
    opt_sim_start = info_fields[2]
    ac_sim_start = info_fields[4]
    rec_start = info_fields[8]
    crop_start = info_fields[10]
    last_line = info_fields[12]

    start_time_date = first_line.split(" ")[0:2]
    start_time = datetime.strptime(first_line.split(" ")[1], "%H:%M:%S,%f")
    vol_creation_start = (datetime.strptime(vol_creation_start.split(" ")[1], "%H:%M:%S,%f") - start_time).total_seconds()
    opt_sim_start = (datetime.strptime(opt_sim_start.split(" ")[1], "%H:%M:%S,%f") - start_time).total_seconds()
    ac_sim_start = (datetime.strptime(ac_sim_start.split(" ")[1], "%H:%M:%S,%f") - start_time).total_seconds()
    rec_start = (datetime.strptime(rec_start.split(" ")[1], "%H:%M:%S,%f") - start_time).total_seconds()
    crop_start = (datetime.strptime(crop_start.split(" ")[1], "%H:%M:%S,%f") - start_time).total_seconds()
    stop_time = (datetime.strptime(last_line.split(" ")[1], "%H:%M:%S,%f") - start_time).total_seconds()

    return {"start_time": start_time_date,
            "vol_creation_start": vol_creation_start,
            "opt_sim_start": opt_sim_start,
            "ac_sim_start": ac_sim_start,
            "rec_start": rec_start,
            "crop_start": crop_start,
            "stop_time": stop_time}

