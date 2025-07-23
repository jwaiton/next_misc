### THE MOTHER SCRIPT
# In principle this script should:
#
# - process events with the relevant corrections file
# - run thekla on the corrected data with relevant changed parameters
# - apply topological cuts and calculate the FOM
# - delete the sophronia data, move files over to /data/halmazan/N100_LPR
#
#
# It should take as an argument
#
# - RUNS (to be processed)
# - BLOBR
# - SCANR
# - VOXEL_SIZE
# - FOM_TS
# 
# and apply them correspondingly
#
# corrections and thekla are run for each ldc file within a bash script
# topology_cut collates the ldcs for each run into a singular file
# FOM then takes them all and creates the FOM

from datetime import datetime
import tables as tb
import pandas as pd
import numpy as np
import json
import glob


import sys,os,os.path
from pathlib import Path
sys.path.append("/scratch/halmazan/NEXT/IC_include-cluster-dropping/IC/")
sys.path.append(os.path.expanduser('~/code/eol_hsrl_python'))
sys.path.append("/scratch/halmazan/NEXT/testing/notebooks/")
os.environ['ICTDIR']='/scratch/halmazan/NEXT/IC_include-cluster-dropping/'

from invisible_cities.types.symbols       import NormStrategy
from invisible_cities.reco.corrections    import read_maps, get_df_to_z_converter, apply_all_correction
from invisible_cities.io.dst_io           import load_dst, load_dsts, df_writer
from invisible_cities.core                import tbl_functions   as tbl
from concurrent.futures import ProcessPoolExecutor


from typing          import Optional
from typing          import Union
from typing          import Callable
from typing          import List
from typing          import Any

from multiprocessing import Pool
from functools import partial


import functions.functions_HE as func
from tqdm import tqdm

# outline all relevant directories
data_dir    = '/data/halmazan/NEXT/N100_LPR/'
scratch_dir = '/scratch/halmazan/NEXT/'
# these are in the pattern 'data/', 'jobs/RN', 'bin'
cut_dir     = f'{scratch_dir}PROCESSING/topology_cuts/'
FOM_dir     = f'{scratch_dir}PROCESSING/FOM/' 
# differ 
corr_dir    = f'{scratch_dir}PROCESSING/corrections/'
corr_out    = f'{scratch_dir}/N100_LPR/' # /RN/CITY/TS/LDC*/*.h5

# thekla differs, 
thekla_dir  = f'{scratch_dir}PROCESSING/thekla_jobs/'
thekla_out  = f'{scratch_dir}/N100_LPR' # /RN/CITY/TS/LDC*/*.h5



def gen_timestamps(run_numbers : List[int]):
    '''
    Generates timestamps for each of the runs provided in a list
    '''
    TIMESTAMPS = []
    for RN in run_numbers:
        TIMESTAMPS.append(datetime.now().strftime('%Y%m%d_%H%M%S_%f'))

    return TIMESTAMPS


def identity(x : Any) -> Any:
    return x


def collect_maps(  map_path   : str
                 , apply_temp : bool
                 , norm_strat : NormStrategy):
    
    maps     = read_maps(os.path.expandvars(map_path))
    get_coef = apply_all_correction( maps
                                   , apply_temp = apply_temp
                                   , norm_strat = norm_strat)
    
    return get_coef


def load_all(f):
        
    dst     = load_dst(f, 'DST',  'Events')
    hits    = load_dst(f, 'RECO', 'Events')
    events  = load_dst(f, 'Run',  'events')
    runInfo = load_dst(f, 'Run',  'runInfo')

    return dst, hits, events, runInfo


def get_calibration_constants(R, MAP2):
    # read in the runs gradient and intercept values
    json_path = f'/scratch/halmazan/NEXT/PROCESSING/corrections/cor_map/{MAP2}'
    with open(json_path, 'r') as file:
        corrections = json.load(file).get(R, {})

    return corrections.get('M', None), corrections.get('C', None)


def correct_martin(df, R, M):
    '''
    Read in the json stored wherever I put it and apply the corrections to energy.
    For each differing peak, correct energy wrt total
    '''

    df = df.copy()

    m, c = get_calibration_constants(R, M)

    ###df['Ec'] = df.Ec * m + c/len(df)
    
    ec_tot = df.Ec.sum()
    ec_cor = (ec_tot * m) + c
    ##ec_diff = ec_cor - ec_tot
    
    ##df['Ec'] += ec_diff * (df['Ec']/ec_tot)
    df['Ec'] = df['Ec'] * (ec_cor/ec_tot)

    return df


def collect_correction_files( R   : int
                            , TS  : int
                            , LDC : int):
    
    # file in and out direction
    files = f'/data/halmazan/NEXT/N100_LPR/{R}/sophronia/prod/ldc{LDC}/'
    print(f'INPUT: {files}')
    files = sorted(glob.glob(files + '*'), key=lambda x: (x.split('/')[-2],int(x.split('/')[-1].split('_')[2])))
    return files



def get_cpus():
    return int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count()))

def set_num_threads_per_process():
    # Get total number of CPUs from SLURM (fallback to os.cpu_count())
    total_cpus = int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count()))

    # You want 7 processes, each with N threads:
    num_processes = 7 # LDC RELATED
    threads_per_process = max(1, total_cpus // num_processes)

    # Set threading environment for each process
    os.environ["OMP_NUM_THREADS"] = str(threads_per_process)
    os.environ["MKL_NUM_THREADS"] = str(threads_per_process)
    os.environ["NUMEXPR_NUM_THREADS"] = str(threads_per_process)
    # (Optional) Log it for debugging
    print(f"SLURM_CPUS_PER_TASK = {total_cpus}")
    print(f"Launching {num_processes} processes with {threads_per_process} threads each")



def load_single_file(file_path):
    '''
    Load data from a single h5 file and produce dataframes for /Tracking/Tracks

    Args:
        file_path       :       str
                                Path to the h5 file to be loaded.

    Returns:
        tracks_df       :       pandas.DataFrame
                                DataFrame containing the /Tracking/Tracks data.
        failed          :       int
                                1 if the file failed to load, 0 otherwise.
    '''
    try: 
        tracks_df = load_dst(file_path, 'Tracking', 'Tracks')
        return tracks_df, 0
    except Exception as e:
        print(f'File {file_path} broke with error:\n{e}')
        x = pd.DataFrame()
        return x, 1

def load_data_fast(folder_path):
    '''
    Load multiple h5 files and produce concatenated dataframes for /Tracking/Tracks, /MC/Particles, and their corresponding eventmap.

    Args:
        folder_path     :       str
                                Path to the folder containing the h5 files.

    Returns:
        tracks          :       pandas.DataFrame
                                Concatenated DataFrame containing the /Tracking/Tracks data from all h5 files.
        total_failures  :       int
                                Total number of failed file loads.
    '''
    
    file_names = [f for f in os.listdir(folder_path) if f.endswith('.h5')]
    file_paths = [os.path.join(folder_path, f) for f in file_names]

    with ProcessPoolExecutor() as executor:
        results = list(executor.map(load_single_file, file_paths))
    
    # Separate the results into respective lists
    tracks_list, failures = zip(*results)

    tracks = pd.concat(tracks_list, axis=0, ignore_index=True)

    # Sum up the failures
    total_failures = sum(failures)

    return tracks, total_failures


def is_last_slurm_task():
                    """
                    Check if this is the last SLURM task running within the job.

                    Returns:
                        bool: True if this is the last SLURM task, False otherwise.
                    """
                    try:
                        job_id = os.environ.get("SLURM_JOB_ID")
                        if not job_id:
                            print("Not running within a SLURM job.")
                            return False

                        # Get the list of running tasks for the current job
                        result = os.popen(f"squeue -j {job_id} -h -o '%T'").read().strip().split("\n")
                        running_tasks = [state for state in result if state == "RUNNING"]
                        print(f'JOBID: {job_id}\n result: {result}\nrunning_tasks: {running_tasks}')
                        # If only one task is running, this is the last task
                        return len(running_tasks) == 1
                    except Exception as e:
                        print(f"Error checking SLURM tasks: {e}")
                        return False

###########################################################################
###########################################################################
###########################################################################
# HIGH LEVEL FUNCTIONS
###########################################################################
###########################################################################
###########################################################################


def correct_and_cut(  files : List[str]
                    , LDC   : int
                    , TS    : int
                    , MAP   : str
                    , MAP2  : str
                    , R     : int):
    '''
    Collect the relevant sophronia files, correct the energy and cut around the DEP
    
    R   -> run number
    TS  -> timestamp
    MAP -> correction map
    LDC -> LDC
    '''

    compression = 'ZLIB4'

    # setup coef
    get_coef = collect_maps( MAP
                           , False
                           , NormStrategy.kr)

    # file in and out direction
    #files = f'/data/halmazan/NEXT/N100_LPR/{R}/sophronia/prod/ldc{LDC}/'
    #print(f'INPUT: {files}')
    #files = sorted(glob.glob(files + '*'), key=lambda x: (x.split('/')[-2],int(x.split('/')[-1].split('_')[2])))
    file_out = f'/scratch/halmazan/NEXT/N100_LPR/{R}/sophronia/{TS}/ldc{LDC}'
    
    if not os.path.exists(file_out):
        print(f"Output folder does not exist: {file_out}. Creating it now.")
        os.makedirs(file_out, exist_ok=True)
    
    passing_dsts = []
    passing_hits = []
    passing_evts = []
    passing_rInf = []

    #for i, f in enumerate(files):
    
    R, I = files.split('/')[-1].split('_')[1:3]

    # Check if the output file already exists
    output_file = f'{file_out}/run_{R}_{I}_ldc{LDC}_{TS}.h5'
    if os.path.exists(output_file):
        print(f'File already exists: {output_file}. Skipping processing.', flush=True)
        return

    # load in
    try:
        dst, hits, events, runInfo = load_all(files)
    except Exception as e:
        print(f'Error loading file: {e}', flush = True)
        
    if 'hits' not in locals() or hits.empty:
        print(f'Empty file, skipping: {files}', flush = True)
        return
        

    # correct
    get_all_coefs = get_coef(hits.X.values, hits.Y.values, hits.Z.values, hits.time.values)
    hits['Ec']    = hits.E * get_all_coefs

    # take only events with corrected energy in the range
    hits = hits[hits.groupby('event').Ec.transform('sum') < 1.9]
    hits = hits[hits.groupby('event').Ec.transform('sum') > 1.4]

    # apply martins correction
    hits = hits.groupby('event', group_keys = False).apply(lambda group: correct_martin(group, R=R, M=MAP2))

    valid_events = hits.event.unique()
    # only keep events in the other dataframes that match
    dst     = dst[dst['event'].isin(valid_events)]
    events  = events[events['evt_number'].isin(valid_events)]
    runInfo = runInfo.head(len(valid_events))

    with tb.open_file(f'{output_file}', 'w', filters = tbl.filters(compression)) as h5out:
        df_writer(h5out,   dst, "DST", "Events" , compression="ZLIB4")
        df_writer(h5out,   hits, "RECO", "Events" , compression="ZLIB4")
        df_writer(h5out,   runInfo, "Run" , "runInfo" , compression="ZLIB4")
        df_writer(h5out,   events, "Run" , "events", compression="ZLIB4")
    


def run_thekla( LDC   : int
              , R     : int
              , TS    : int
              , blobR : int
              , scanR : int
              , voxS  : int):
    
    '''
    run the bash script across the thekla process
    '''

    
    # then run thekla
    # Construct the bash command for running thekla
    thekla_script = f"{scratch_dir}PROCESSING/full_monty/bin/run_thekla.sh"
    thekla_command = f"bash {thekla_script} {R} {TS} {blobR} {scanR} {voxS} {LDC}"

    print(f'Running thekla with command: {thekla_command}', flush=True)

    # Execute the bash command
    exit_code = os.system(thekla_command)
    if exit_code != 0:
        print(f'Thekla script failed for RUN {R} with exit code {exit_code}', flush=True)



def cut_and_save( R       : int
                , TS      : int
                , z_lower : float
                , z_upper : float
                , r_lim   : float
                , e_lower : float
                , e_upper : float
                , city    : str
                , blobR   : int
                , scanR   : int
                , voxelS  : int):


    print(f'R-{R}, TS-{TS}\nblobR: {blobR}\nscanR: {scanR}\nvoxelS: {voxelS}')
    
    root_path = '/scratch/halmazan/NEXT/'

    # get a directory to save to
    folder_name = f'{root_path}PROCESSING/topology_cuts/data/{R}/bR{blobR}_sR{scanR}_vS{voxelS}/{TS}'
    folder_s = Path(f'{folder_name}')
    folder_s.mkdir(parents=True, exist_ok=True)

    # load in
    n100_dir = f'{root_path}N100_LPR/{R}/{city}/{TS}/'

    hdst = []
    errors = 0
    for i in tqdm(range(1,8)):
        print(f"LDC {i}")
        folder_path = n100_dir + 'ldc' + str(i) + '/'
        holder, err = load_data_fast(folder_path)
        r = holder
        errors += err
        hdst.append(r)
    hdst = pd.concat(hdst)
    
    print('=' * 20)
    print(f'Number of failed files: {errors}')
    cut_hdst, efficiencies = func.apply_cuts(hdst, 
                                             lower_z = z_lower, 
                                             upper_z = z_upper, 
                                             r_lim   = r_lim, 
                                             lower_e = e_lower, 
                                             upper_e = e_upper)
    print('=' * 20)
    print(efficiencies)

    # SAVE THE DATAFRAME
    cut_hdst.to_hdf(f'{folder_name}/cut_hdst.h5', key = 'Tracking/Tracks')

def resolve_data(RUNS, TIMESTAMPS):
    '''
    will remove all the relevant data from:
    - sophronia
    and move thekla data to $DATA
    '''
    # data from correct and cut:
    for rn, ts in zip(RUNS, TIMESTAMPS):
        folder_out = f'/scratch/halmazan/NEXT/N100_LPR/{rn}/sophronia/{ts}/'
        # Recursively delete everything in folder_out
        try:
            if os.path.exists(folder_out):
                for root, dirs, files in os.walk(folder_out, topdown=False):
                    for file in files:
                        os.remove(os.path.join(root, file))
                    for dir in dirs:
                        os.rmdir(os.path.join(root, dir))
                os.rmdir(folder_out)
                print(f"Successfully deleted folder: {folder_out}")
            else:
                print(f"Folder does not exist: {folder_out}")
        except Exception as e:
            print(f"Error while deleting folder {folder_out}: {e}")


###########################################################################
###########################################################################
###########################################################################
# MAIN
###########################################################################
###########################################################################
###########################################################################


def main( RUNS       : List[int]
        , blobR      : int
        , scanR      : int
        , voxel_size : int
        , FOM_TS     : int
        , MAP        : str
        , MAP2       : str
        , topo_cuts  : List
        , LDC        : int
        , TIMESTAMP  : str):



    #set_num_threads_per_process()

    # Correction map defined here
    corr_map    = f'{scratch_dir}PROCESSING/corrections/cor_map/{MAP}'

    # generate timestamp for all the runs, they can match as they dont overlap!
    TIMESTAMPS = [TIMESTAMP] * len(RUNS)

    print('=' * 20)
    

    print('=' * 20)
    print('APPLYING CUTS...')
    print('=' * 20)

    print(f'RUNS AND TIMESTAMPS')
    print('=' * 20)
    for RN, TS in zip(RUNS, TIMESTAMPS):
        print(f'=  {RN}: {TS}  =')

        # get list of file paths
        files = collect_correction_files(RN, TS, LDC)
        cpus = get_cpus()
        print(f'Number of available CPUS within this task: {cpus}')
        print(f'This is just a crosscheck!')
        # pool per cpus (should be normal)
        
        
        process_partial = partial(correct_and_cut, R=RN, LDC=LDC, TS=TS, MAP=corr_map, MAP2=MAP2)
        with Pool(processes=cpus) as pool:  # Adjust the number of processes as needed
            pool.map(process_partial, files)

        ####run_thekla(LDC=LDC, R=RN, TS=TS, blobR=blobR, scanR=scanR, voxS=voxel_size)

        #process_thekla = partial(run_thekla, R=RN, TS=TS, blobR=blobR, scanR=scanR, voxS=voxel_size)
        #with Pool(processes=7) as pool:  # Adjust the number of processes as needed
        #    pool.map(process_thekla, range(1, 2))
        #break

        # non parallel correct and cut (bad times)
        #correct_and_cut(LDC=LDC, R=RN, TS=TS, MAP=corr_map, MAP2=MAP2)

    # if you're the last slurm task, you should have the responsiblity of compiling the LDCs
    ####for RN, TS in zip(RUNS, TIMESTAMPS):       
    ####
    ####    if is_last_slurm_task():
    ####        cut_and_save(R=RN, TS=TS, 
    ####                     z_lower=topo_cuts['z_lower'], 
    ####                     z_upper=topo_cuts['z_upper'], 
    ####                     r_lim=topo_cuts['r_lim'], 
    ####                     e_lower=topo_cuts['e_lower'], 
    ####                     e_upper=topo_cuts['e_upper'], 
    ####                     city='thekla', 
    ####                     blobR=blobR, 
    ####                     scanR=scanR, 
    ####                     voxelS=voxel_size)
    ####
    ####        # remove the useless data also
    ####        resolve_data(RUNS, TIMESTAMPS)
    ####    else:
    ####        print(f'ldc{LDC} is not the last slurm task!')
                


                

if __name__ == '__main__':
    
    for args in sys.argv:
        print(args)    

    RUNS          = sys.argv[1]
    RUNS          = [int(x) for x in RUNS.split(',')]
    blobR         = int(sys.argv[2])
    scanR         = int(sys.argv[3])
    voxel_size    = int(sys.argv[4])
    FOM_TS        = int(sys.argv[5])
    MAP           = sys.argv[6]
    SECONDARY_MAP = sys.argv[7]
    print('topology cuts')
    print(repr(sys.argv[8]))
    topo_cuts     = json.loads(sys.argv[8])
    LDC           = sys.argv[9]
    TIMESTAMP     = str(sys.argv[10])

    main(RUNS, blobR, scanR, voxel_size, FOM_TS, MAP, SECONDARY_MAP, topo_cuts, LDC, TIMESTAMP)