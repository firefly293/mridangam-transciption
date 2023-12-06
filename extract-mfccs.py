import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt

def extract_mfcc(wav_file_path: str, start_time: float, end_time: float, n_mfcc: int = 13):
    '''Extracts the MFCCs from an audio file with path `wave_file_path` from time `start_time` to time `end_time`'''
    # get the duration
    duration = librosa.get_duration(path=wav_file_path)

    # ensure input is valid
    if (start_time >= end_time or start_time < 0 or end_time <= 0 or start_time >= duration or end_time > duration):
        return None
    
    # load the file
    y, sr = librosa.load(wav_file_path, sr=44100)

    # trim to desired time frame
    startSample = int(start_time * sr)
    endSample = int(end_time * sr)
    y = y[startSample:endSample]

    # extract the MFCCs
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    print(mfccs.shape)

    return mfccs


def plot_mfcc(mfccs):
    # Display the MFCCs
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(mfccs, x_axis='time', cmap='viridis')
    plt.title('MFCC')
    plt.show()


wav_file = 'resources/data/advait-data-01/advait-data-01.wav'
start_timestamp = 1.0 
end_timestamp = 5.0    


mfccs = extract_mfcc(wav_file, start_timestamp, end_timestamp)


plot_mfcc(mfccs)
    

    
