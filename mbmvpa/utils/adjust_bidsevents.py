from bids import BIDSLayout
import pandas as pd
from .events_utils import _preprocess_event
import tqdm

def adjust_save(root,
               task_name=None,
               adjust_function=lambda x: x,
               filter_function=lambda _: True,
               column_names =None):
    
    layout = BIDSLayout(root=root)
    events_files = layout.get(task=task_name,
                              suffix='events',
                              extension="tsv")
    
    for events_file in tqdm.tqdm(events_files):
        df_events = pd.read_table(events_file.path)
        df_events = _preprocess_event(adjust_function, df_events)
        df_events= pd.concat([df_events[[filter_function(row) \
                                for _, row in df_events.iterrows()]]])
        
        if column_names is not None:
            df_events = df_events[column_names]
        df_events.to_csv(events_file.path, index=False, sep="\t")
    
    
