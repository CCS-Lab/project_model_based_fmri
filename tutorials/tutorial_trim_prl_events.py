from mbmvpa.utils.adjust_bidsevents import adjust_save

root = 'tutorial_data/ccsl_prl'


def example_adjust(row):
    if row["outcome"] == 0:
        row["outcome"] = -1
    row["onset"] = row["time_onset"]
    row["duration"] = 1 #row["time_wait"] - row["time_feedback"]
    return row

def example_filter(row):
    return row['choice'] in [1,2]

adjust_save(root,
           task_name='prl',
           adjust_function=example_adjust,
           filter_function=example_filter)