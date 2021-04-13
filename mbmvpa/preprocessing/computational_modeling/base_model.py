
class Base():
    def __init__(self, process_name):
        self.process_name = process_name
        self.latent_process = {}
    
    def _set_latent_process(self, df_events, param_dict):
        # implement
        return
    
    def _add(self, key, value):
        if key not in self.latent_process.keys():
            self.latent_process[key] = []
        self.latent_process[key].append(value)
    
    def __call__(self, df_events, param_dict):
        self._set_latent_process(df_events, param_dict)
        df_events["modulation"] = self.latent_process[self.process_name]
        return df_events[['onset','duration','modulation']]
        