from bids import BIDSLayout
import pandas as pd
from .events_utils import _make_function_dfwise
import tqdm

def adjust_save(root,
               task_name=None,
               adjust_function=lambda x: x,
               filter_function=lambda _: True,
               adjust_function_dfwise=None,
               filter_function_dfwise=None,
               column_names =None):
    
    layout = BIDSLayout(root=root)
    events_files = layout.get(task=task_name,
                              suffix='events',
                              extension="tsv")
    
    if adjust_function_dfwise is None:
        adjust_function_dfwise = _make_function_dfwise(adjust_function)

    if filter_function_dfwise is None:
        filter_function_dfwise = _make_function_dfwise(filter_function)

    for events_file in tqdm.tqdm(events_files):
        df_events = pd.read_table(events_file.path)
        df_events = filter_function_dfwise(
                                adjust_function_dfwise(df_events))
        
        df_events= pd.concat([df_events[[filter_function(row) \
                                for _, row in df_events.iterrows()]]])
        
        if column_names is not None:
            df_events = df_events[column_names]
        df_events.to_csv(events_file.path, index=False, sep="\t")
    
    
