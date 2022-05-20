import json


def print_stage(stage_str):
    count = 100
    occupied_count = len(stage_str)
    separator_num = int((count - occupied_count) / 2)
    separator_str = "=" * separator_num
    print_str = f"{separator_str}{stage_str}{separator_str}"
    print(print_str)


def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


# save to the file
def dict_to_json(data_dict, file_name):
    with open(file_name, "w") as outfile:
        json.dump(data_dict, outfile, indent=4)
