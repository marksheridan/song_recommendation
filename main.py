# This is a sample Python script.

import wave

import pandas as pd
import numpy as np
import librosa
import time
import os
import matplotlib.pyplot as plt
from pydub import AudioSegment
from pydub.playback import play


def check_values(row):
    result_row = []
    for val in row:
        if val < 0.8:
            result_row.append(0)
        else:
            result_row.append(1)

    return result_row


def check_chromagram(chromagram):
    result = []
    for row in chromagram:
        result.append(check_values(row))
    return result


def calculate_row_value(row):
    return (sum(row) / len(row))


def get_chroma_vector_value(chromagram):
    res = np.array(check_chromagram(chromagram))
    output = []
    for i in range(0, res.shape[0]):
        val = calculate_row_value(res[i])
        output.append(val)
    return output


# file = '33231672_Moina-Tumi-Jui.mp3'
#
# raw_file = AudioSegment.from_file(file="./data/"+file, format="mp3")
# f_name = file.replace(".mp3",".wav")
# raw_file.export(out_f="./export/"+f_name, format="wav")


# fig, ax = plt.subplots(figsize=(15, 3))
# img = librosa.display.specshow(chromagram, x_axis='time', y_axis='chroma', hop_length=hop_length, cmap='coolwarm')
# fig.colorbar(img, ax=ax)




def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


def get_audio_segment_info(audio_segment, song):
    row = []
    row.append(song)
    row.append(audio_segment.channels)
    row.append(audio_segment.sample_width)
    row.append(audio_segment.frame_rate)
    row.append(audio_segment.frame_width)
    row.append(len(audio_segment))
    row.append(audio_segment.frame_count())
    row.append(audio_segment.dBFS)
    return row

if __name__ == '__main__':

    cols = ['File name', 'Channels', 'Sample width', 'Frame rate (sample rate)', 'Frame width', 'Length (ms)',
               'Frame count', 'Intensity', 'C', 'C♯', 'D', 'D♯', 'E' , 'F', 'F♯', 'G', 'G♯', 'A', 'A♯', 'B']
    print_hi('PyCharm')
    hop_length = 512
    song_list = os.listdir('./data')
    df_data = []
    for file in os.listdir('./data'):
        raw_file = AudioSegment.from_file(file="./data/" + file, format="mp3")
        f_name = file.replace(".mp3", ".wav")
        raw_file.export(out_f="./export/" + f_name, format="wav")


    for song in song_list:
        start_time = time.time()
        audio_segment = AudioSegment.from_file("./data/" + song)
        x, sr = librosa.load('./data/' + song)
        row = get_audio_segment_info(audio_segment, song)
        chromagram = librosa.feature.chroma_stft(x, sr=sr, hop_length=hop_length)
        chroma_vector_data = get_chroma_vector_value(chromagram)
        print(chroma_vector_data)
        print("--- %s seconds ---" % (time.time() - start_time))
        row = row + chroma_vector_data
        df_data.append(row)

        print("*"*40)
    audio_info_df = pd.DataFrame(df_data, columns= cols)

    audio_info_df.to_csv('test_result.csv', index=False)
# See PyCharm help at https://www.jetbrains.com/help/pycharm/
