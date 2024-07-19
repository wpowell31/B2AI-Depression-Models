import matplotlib.pyplot as plt
import IPython.display as Ipd
import pandas as pd
import torch
from pathlib import Path
import numpy as np
from b2aiprep.process import SpeechToText
from b2aiprep.process import Audio, specgram
from b2aiprep.dataset import VBAIDataset
from b2aiprep.process import extract_opensmile



# Load the b2ai dataset
path = '/home/bridge2ai/Desktop/bridge2ai-data/bids_with_sensitive_recordings/'
dataset = VBAIDataset(path)

# load and clean participant and demographics dataframes
participant_df = dataset.load_and_pivot_questionnaire('participant')
demographics_df = dataset.load_and_pivot_questionnaire('qgenericdemographicsschema')
demographics_df_clean = demographics_df[demographics_df.duplicated('record_id', keep=False) == False]
participant_df_clean = participant_df[participant_df.duplicated('record_id', keep=False) == False]

# merge cleaned participant and demographic dataframes
# Perform an inner join on 'record_id'
participant_demo_df = pd.merge(
   demographics_df_clean, 
   participant_df_clean, 
   on='record_id', 
   how='inner'
)

participant_demo_df = pd.DataFrame(list(participant_demo_df.values), columns=list(participant_demo_df.columns))
participant_demo_df.to_csv('participant_demo_df.csv', index=False)

# Load recording data
recording_df = dataset._load_recording_and_acoustic_task_df()

def extract_all_audio_type(recording_df, dataset, recording_name):
   """Extract all audio recordings that are of the same task
   Input: filtered recording_df with recordings of same task
   Output: nested dictionary with record_id and recording_ig and audio data
   Think about what to do if there are duplicates - need to check
   record_ids should be filtered in as unique"""
   filtered_recordings = recording_df[recording_df["recording_name"] == recording_name]
   # For now - get only the record_ids that appear only once for each speech type
   # Step 1: Identify duplicates
   duplicates = filtered_recordings['record_id'].duplicated(keep=False)

   # Step 2: Filter out duplicates
   filtered_recordings = filtered_recordings[~duplicates]
   record_ids = filtered_recordings['record_id']
   GeMAPS_audio_features = pd.DataFrame(
       columns=extract_opensmile(dataset.load_recording("30F10F81-5B89-4103-AB2A-CDE4F4DFB9F9")).columns
   )
   i=0
   for record_id in record_ids:
       # ge row of that record_id
       idx = recording_df['record_id'] == record_id
       row = filtered_recordings.loc[idx].iloc[0]
       recording_id = row['recording_id']
       audio = dataset.load_recording(recording_id)
       GeMAPS_features = extract_opensmile(audio)
       GeMAPS_features['record_id'] = record_id
       GeMAPS_audio_features = pd.concat(
           [GeMAPS_audio_features, GeMAPS_features],
           ignore_index = True
       )
       i += 1
       print(f'Loaded subject {i}')
       
   return GeMAPS_audio_features

GeMAPS_audio_features1 = extract_all_audio_type(recording_df,
                        dataset,
                        "Free speech-1")

GeMAPS_audio_features2 = extract_all_audio_type(recording_df,
                        dataset,
                        "Free speech-2")

GeMAPS_audio_features3 = extract_all_audio_type(recording_df,
                        dataset,
                        "Free speech-3")

GeMAPS_audio_features = pd.concat([GeMAPS_audio_features1, GeMAPS_audio_features2, GeMAPS_audio_features3]).groupby('record_id').mean().reset_index()

# Combine with demographic data
GeMAPS_demo_df = pd.merge(
    GeMAPS_audio_features,
    participant_demo_df,
    on='record_id',
    how='left'
)

# Filter out selected columns
GeMAPS_demo_df = GeMAPS_demo_df[[
   'record_id', 'age', 'osa', 'depression', 'soc_anx_dis', 'gad', 'add_adhd', 'asd', 'bipolar', 'bpd', 'ed', 'insomnia', 'ocd', 
   'panic', 'ptsd', 'schizophrenia', 'other_psych', 'asthma', 'airway_stenosis', 'chronic_cough', 'copd', 'laryng_cancer', 
   'benign_cord_lesion', 'rrp', 'spas_dys', 'voc_fold_paralysis', 'alz_dementia_mci', 'als', 'parkinsons', 'alcohol_subst_abuse',
   'F0semitoneFrom27.5Hz_sma3nz_amean', 'F0semitoneFrom27.5Hz_sma3nz_stddevNorm','F0semitoneFrom27.5Hz_sma3nz_percentile20.0',
   'F0semitoneFrom27.5Hz_sma3nz_percentile50.0','F0semitoneFrom27.5Hz_sma3nz_percentile80.0', 'F0semitoneFrom27.5Hz_sma3nz_pctlrange0-2',
   'F0semitoneFrom27.5Hz_sma3nz_meanRisingSlope','F0semitoneFrom27.5Hz_sma3nz_stddevRisingSlope','F0semitoneFrom27.5Hz_sma3nz_meanFallingSlope',
   'F0semitoneFrom27.5Hz_sma3nz_stddevFallingSlope','loudness_sma3_amean','loudness_sma3_stddevNorm','loudness_sma3_percentile20.0','loudness_sma3_percentile50.0',
   'loudness_sma3_percentile80.0','loudness_sma3_pctlrange0-2','loudness_sma3_meanRisingSlope','loudness_sma3_stddevRisingSlope','loudness_sma3_meanFallingSlope',
   'loudness_sma3_stddevFallingSlope','spectralFlux_sma3_amean','spectralFlux_sma3_stddevNorm','mfcc1_sma3_amean','mfcc1_sma3_stddevNorm','mfcc2_sma3_amean',
   'mfcc2_sma3_stddevNorm','mfcc3_sma3_amean','mfcc3_sma3_stddevNorm','mfcc4_sma3_amean','mfcc4_sma3_stddevNorm','jitterLocal_sma3nz_amean','jitterLocal_sma3nz_stddevNorm',
   'shimmerLocaldB_sma3nz_amean','shimmerLocaldB_sma3nz_stddevNorm','HNRdBACF_sma3nz_amean','HNRdBACF_sma3nz_stddevNorm','logRelF0-H1-H2_sma3nz_amean',
   'logRelF0-H1-H2_sma3nz_stddevNorm','logRelF0-H1-A3_sma3nz_amean','logRelF0-H1-A3_sma3nz_stddevNorm','F1frequency_sma3nz_amean','F1frequency_sma3nz_stddevNorm',
   'F1bandwidth_sma3nz_amean','F1bandwidth_sma3nz_stddevNorm','F1amplitudeLogRelF0_sma3nz_amean','F1amplitudeLogRelF0_sma3nz_stddevNorm','F2frequency_sma3nz_amean',
   'F2frequency_sma3nz_stddevNorm','F2bandwidth_sma3nz_amean','F2bandwidth_sma3nz_stddevNorm','F2amplitudeLogRelF0_sma3nz_amean','F2amplitudeLogRelF0_sma3nz_stddevNorm',
   'F3frequency_sma3nz_amean','F3frequency_sma3nz_stddevNorm','F3bandwidth_sma3nz_amean','F3bandwidth_sma3nz_stddevNorm','F3amplitudeLogRelF0_sma3nz_amean',
   'F3amplitudeLogRelF0_sma3nz_stddevNorm','alphaRatioV_sma3nz_amean','alphaRatioV_sma3nz_stddevNorm','hammarbergIndexV_sma3nz_amean','hammarbergIndexV_sma3nz_stddevNorm',
   'slopeV0-500_sma3nz_amean','slopeV0-500_sma3nz_stddevNorm','slopeV500-1500_sma3nz_amean','slopeV500-1500_sma3nz_stddevNorm','spectralFluxV_sma3nz_amean',
   'spectralFluxV_sma3nz_stddevNorm','mfcc1V_sma3nz_amean','mfcc1V_sma3nz_stddevNorm','mfcc2V_sma3nz_amean','mfcc2V_sma3nz_stddevNorm','mfcc3V_sma3nz_amean',
   'mfcc3V_sma3nz_stddevNorm','mfcc4V_sma3nz_amean','mfcc4V_sma3nz_stddevNorm','alphaRatioUV_sma3nz_amean','hammarbergIndexUV_sma3nz_amean','slopeUV0-500_sma3nz_amean','slopeUV500-1500_sma3nz_amean',
   'spectralFluxUV_sma3nz_amean','loudnessPeaksPerSec','VoicedSegmentsPerSec','MeanVoicedSegmentLengthSec','StddevVoicedSegmentLengthSec','MeanUnvoicedSegmentLength',
   'StddevUnvoicedSegmentLength','equivalentSoundLevel_dBp']]

# Make Columns numeric
GeMAPS_demo_df.replace({True: 1, False: 0}, inplace=True)
GeMAPS_demo_df['age'] = GeMAPS_demo_df['age'].astype('float')

# Drop columns not used for machine learning
if 'Unnamed: 0' in GeMAPS_demo_df.columns:
   GeMAPS_demo_df.drop(columns=['Unnamed: 0', 'record_id'])
else:
   GeMAPS_demo_df.drop(columns=['record_id'])


# drop rows with missing outcome values
GeMAPS_demo_df = GeMAPS_demo_df.dropna(subset=['depression'])

# Drop other anxiety related columns - avoid leakage
GeMAPS_demo_df = GeMAPS_demo_df[~((GeMAPS_demo_df['depression'] == 0) &
                                  ((GeMAPS_demo_df['gad'] == 1) | 
                                  (GeMAPS_demo_df['panic'] == 1) | 
                                  (GeMAPS_demo_df['soc_anx_dis'] == 1) |
                                    (GeMAPS_demo_df['other_psych'] == 1) |
                                    (GeMAPS_demo_df['panic'] == 1) |
                                    (GeMAPS_demo_df['ptsd'] == 1) |
                                    (GeMAPS_demo_df['schizophrenia'] == 1)
                                  ))]

# Drop 'gad', 'panic', 'soc_anx_dis' columns
GeMAPS_demo_df = GeMAPS_demo_df.drop(columns=['gad', 'panic', 'soc_anx_dis', 'other_psych', 'panic', 'ptsd', 'schizophrenia'])

# save the dataframe to a csv file
GeMAPS_demo_df.to_csv('GeMAPS_demo_df2.csv')