import pandas as pd
from collections import defaultdict

def create_cell_report(signal, voxel, output_path, structure_path = './structures.csv', target_id = None):
    """
    Create an Excel report from aggregated regional statistics.

    Parameters:
        signal (dict): Dictionary of region stats from numba_unique_vessel.
        voxel (tuple): Physical voxel size (Z, Y, X) for mm^3 conversion.
        output_path (str): Path to save the Excel report.
        structure_path (str): CSV file path of brain structure hierarchy.
        target_id (list): Specific structure IDs to include in a special summary sheet.
    """
    list_signal = [[key] + list(value) for key, value in signal.items()]
    
    with pd.ExcelWriter(output_path) as writer:
        structure = pd.read_csv(structure_path)
        
        summary = pd.DataFrame(list_signal, columns=['id', 'total_volume', 'total_cells'])
        summary = pd.merge(structure, summary, on='id', how='left').fillna(0)
        summary.set_index('id', inplace = True)
        
        tiers = defaultdict(list)
        for path_id in structure['structure_id_path']:
            parts = path_id.strip('/').split('/')
            tiers[len(parts)].append(int(parts[-1]))
        tiers = dict(sorted(tiers.items(), reverse=True))

        for tier, ids in tiers.items():
            sheet = defaultdict(list)
            for id in ids:
                temp = summary.loc[summary['parent_structure_id'].isin([id])]
                
                summary.at[id, 'total_volume'] += temp['total_volume'].sum()
                summary.at[id, 'total_cells'] += temp['total_cells'].sum()

                sheet['structure_name'].append(summary.at[id, 'name'])
                sheet['acronym'].append(summary.at[id, 'acronym'])
                sheet['total_cells'].append(summary.at[id, 'total_cells']) 
                sheet['total_volume'].append(summary.at[id, 'total_volume'])

            sheet = pd.DataFrame(sheet) #.sort_values(by=['total_cells'], ascending=False)
            sheet = sheet[['structure_name', 'acronym', 'total_volume', 'total_cells']]
            sheet['total_volume_mm3'] = sheet['total_volume'] * voxel
            sheet['total_cell_density'] = sheet['total_cells']/sheet['total_volume']
            sheet.fillna(0).to_excel(writer, sheet_name=f"Tier {tier}", index=False)
        
        if target_id is not None:
            sheet = defaultdict(list)
            for id in target_id:
                sheet['structure_name'].append(summary.at[id, 'name'])
                sheet['acronym'].append(summary.at[id, 'acronym'])
                sheet['total_cells'].append(summary.at[id, 'total_cells']) 
                sheet['total_volume'].append(summary.at[id, 'total_volume'])
                
            sheet = pd.DataFrame(sheet) #.sort_values(by=['total_cells'], ascending=False)
            sheet = sheet[['structure_name', 'acronym', 'total_volume', 'total_cells']]
            sheet['total_volume_mm3'] = sheet['total_volume'] * voxel
            sheet['total_cell_density'] = sheet['total_cells']/sheet['total_volume']
            sheet.fillna(0).to_excel(writer, sheet_name=f"Target Region", index=False)

def create_vessel_report(signal, voxel, output_path, structure_path = './structures.csv', target_id = None):
    """
    Create an Excel report from aggregated regional statistics.

    Parameters:
        signal (dict): Dictionary of region stats from numba_unique_vessel.
        voxel (tuple): Physical voxel size (Z, Y, X) for mm^3 conversion.
        output_path (str): Path to save the Excel report.
        structure_path (str): CSV file path of brain structure hierarchy.
        target_id (list): Specific structure IDs to include in a special summary sheet.
    """
    list_signal = [[key] + list(value) for key, value in signal.items()]
    
    with pd.ExcelWriter(output_path) as writer:
        structure = pd.read_csv(structure_path)
        
        summary = pd.DataFrame(list_signal, columns=[
            'id', 'total_volume', 'total_signal_volume', 'total_skeleton_volume', 
            'total_bifurcations_count', 'total_radius_amount', 'max_radius'
        ])
        summary = pd.merge(structure, summary, on='id', how='left').fillna(0)
        summary.set_index('id', inplace = True)
        
        tiers = defaultdict(list)
        for path_id in structure['structure_id_path']:
            parts = path_id.strip('/').split('/')
            tiers[len(parts)].append(int(parts[-1]))
        tiers = dict(sorted(tiers.items(), reverse=True))

        for tier, ids in tiers.items():
            sheet = defaultdict(list)
            for id in ids:
                temp = summary.loc[summary['parent_structure_id'].isin([id])]
                
                summary.at[id, 'total_volume'] += temp['total_volume'].sum()
                summary.at[id, 'total_signal_volume'] += temp['total_signal_volume'].sum()
                summary.at[id, 'total_skeleton_volume'] += temp['total_skeleton_volume'].sum()
                summary.at[id, 'total_bifurcations_count'] += temp['total_bifurcations_count'].sum()
                summary.at[id, 'total_radius_amount'] += temp['total_radius_amount'].sum()
                if pd.isna(summary.at[id, 'max_radius']):
                    summary.at[id, 'max_radius'] = temp['max_radius'].max()
                else:
                    summary.at[id, 'max_radius'] = max(summary.at[id, 'max_radius'], temp['max_radius'].max())

                sheet['structure_name'].append(summary.at[id, 'name'])
                sheet['acronym'].append(summary.at[id, 'acronym'])
                sheet['total_volume'].append(summary.at[id, 'total_volume'])
                sheet['total_signal_volume'].append(summary.at[id, 'total_signal_volume'])
                sheet['total_skeleton_volume'].append(summary.at[id, 'total_skeleton_volume'])
                sheet['total_bifurcations_count'].append(summary.at[id, 'total_bifurcations_count'])
                sheet['total_radius_amount'].append(summary.at[id, 'total_radius_amount'])
                sheet['max_radius'].append(summary.at[id, 'max_radius'])
 
            sheet = pd.DataFrame(sheet) #.sort_values(by=['total_bifurcations_count'], ascending=False)
            sheet = sheet[[
                'structure_name', 'acronym', 'total_volume', 'total_signal_volume', 
                'total_skeleton_volume', 'total_bifurcations_count', 'total_radius_amount', 'max_radius'
            ]]
            
            sheet['total_volume_mm3'] = sheet['total_volume'] * voxel
            sheet['total_signal_volume_mm3'] = sheet['total_signal_volume'] * voxel
            sheet['total_skeleton_volume_mm3'] = sheet['total_skeleton_volume'] * voxel
            
            sheet['total_signal_density'] = sheet['total_signal_volume'] / sheet['total_volume']
            sheet['total_skeleton_density'] = sheet['total_skeleton_volume'] / sheet['total_volume']
            sheet['total_bifurcations_density'] = sheet['total_bifurcations_count'] / sheet['total_volume']
            sheet['mean_radius'] = sheet['total_radius_amount'] / sheet['total_signal_volume']
            
            sheet.fillna(0).to_excel(writer, sheet_name=f"Tier {tier}", index=False)
        
        if target_id is not None:
            sheet = defaultdict(list)
            for id in target_id:
                sheet['structure_name'].append(summary.at[id, 'name'])
                sheet['acronym'].append(summary.at[id, 'acronym'])
                sheet['total_volume'].append(summary.at[id, 'total_volume'])
                sheet['total_signal_volume'].append(summary.at[id, 'total_signal_volume'])
                sheet['total_skeleton_volume'].append(summary.at[id, 'total_skeleton_volume'])
                sheet['total_bifurcations_count'].append(summary.at[id, 'total_bifurcations_count'])
                sheet['total_radius_amount'].append(summary.at[id, 'total_radius_amount'])
                sheet['max_radius'].append(summary.at[id, 'max_radius'])
                
            sheet = pd.DataFrame(sheet) #.sort_values(by=['total_bifurcations_count'], ascending=False)
            sheet = sheet[[
                'structure_name', 'acronym', 'total_volume', 'total_signal_volume', 
                'total_skeleton_volume', 'total_bifurcations_count', 'total_radius_amount', 'max_radius'
            ]]
            
            sheet['total_volume_mm3'] = sheet['total_volume'] * voxel
            sheet['total_signal_volume_mm3'] = sheet['total_signal_volume'] * voxel
            sheet['total_skeleton_volume_mm3'] = sheet['total_skeleton_volume'] * voxel
            
            sheet['total_signal_density'] = sheet['total_signal_volume'] / sheet['total_volume']
            sheet['total_skeleton_density'] = sheet['total_skeleton_volume'] / sheet['total_volume']
            sheet['total_bifurcations_density'] = sheet['total_bifurcations_count'] / sheet['total_volume']
            sheet['mean_radius'] = sheet['total_radius_amount'] / sheet['total_signal_volume']
                
            sheet.fillna(0).to_excel(writer, sheet_name=f"Target Region", index=False)