# Copyright 2019 Aptiv
# Copyright 2021 Motional
# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
from types import ModuleType
from typing import Dict, List


def nuscenes_mot_metric(nusceneseval: ModuleType, options: Dict):
    @staticmethod
    def merge_event_dataframes_new(dfs, update_frame_indices=True, update_oids=True, update_hids=True, 
                                   return_mappings=False): 
        mapping_infos = []
        new_oid = count()
        new_hid = count()

        r = MOTAccumulatorCustom.new_event_dataframe()
        for df in dfs:

            if isinstance(df, MOTAccumulatorCustom):
                df = df.events

            copy = df.copy()
            infos = {}

            # Update index
            if update_frame_indices:
                if r.index.get_level_values(0).size > 0 and isinstance(r.index.get_level_values(0)[0], tuple):
                    index_temp = []
                    for item in r.index.get_level_values(0):
                        index_temp += list(item)
                    index_temp = np.array(index_temp)
                    index_temp_unique = np.unique(index_temp)
                    next_frame_id = max(np.max(index_temp) + 1, index_temp_unique.shape[0])
                else:                
                    next_frame_id = max(r.index.get_level_values(0).max() + 1,
                                        r.index.get_level_values(0).unique().shape[0])

                if np.isnan(next_frame_id):
                    next_frame_id = 0
                copy.index = copy.index.map(lambda x: (x[0] + next_frame_id, x[1]))
                infos['frame_offset'] = next_frame_id

            # Update object / hypothesis ids
            if update_oids:
                oid_map = dict([oid, str(next(new_oid))] for oid in copy['OId'].dropna().unique())
                copy['OId'] = copy['OId'].map(lambda x: oid_map[x], na_action='ignore')
                infos['oid_map'] = oid_map

            if update_hids:
                hid_map = dict([hid, str(next(new_hid))] for hid in copy['HId'].dropna().unique())
                copy['HId'] = copy['HId'].map(lambda x: hid_map[x], na_action='ignore')
                infos['hid_map'] = hid_map

            r = pd.concat((r, copy))
            mapping_infos.append(infos)

        if return_mappings:
            return r, mapping_infos
        else:
            return r
    
    if hasattr(nusceneseval, "mot"):
        nusceneseval.mot.MOTAccumulatorCustom.merge_event_dataframes = merge_event_dataframes_new