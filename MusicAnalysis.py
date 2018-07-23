from random import shuffle
import librosa
import librosa.display
import IPython
import numpy as np
import pandas as pd
import scipy
import matplotlib.pyplot as plt
import seaborn as sns
import glob


files = []
for file in list(glob.glob("D:\\My Jobs\\Job 4\\wav\\*")):
    files += list(glob.glob(file + "\\*.wav"))
shuffle(files)

octave = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
with open("music_ext.csv", "w") as txt:
    txt.writelines("name,mood,duration,sample_total," +
                   # "mfcc_mean_0,mfcc_std_0,mfcc_mean_1,mfcc_std_1," +
                   # "mfcc_mean_2,mfcc_std_2,mfcc_mean_3,mfcc_std_3," +
                   # "mfcc_mean_4,mfcc_std_4,mfcc_mean_5,mfcc_std_5," +
                   # "mfcc_mean_6,mfcc_std_6,mfcc_mean_7,mfcc_std_7," +
                   # "mfcc_mean_8,mfcc_std_8,mfcc_mean_9,mfcc_std_9," +
                   # "mfcc_mean_10,mfcc_std_10,mfcc_mean_11,mfcc_std_11," +
                   # "mfcc_mean_12,mfcc_mean_12,cent_mean,cent_std,cent_skew," +
                   # "contr_mean_0,contr_std_0,contr_mean_1,contr_std_1," +
                   # "contr_mean_2,contr_std_2,contr_mean_3,contr_std_3," +
                   # "contr_mean_4,contr_std_4,contr_mean_5,contr_std_5," +
                   # "contr_mean_6,contr_std_6,roll_mean,roll_std,roll_skew," +
                   "c_mean,c_sharp_mean,d_mean,d_sharp_mean,e_mean,f_mean," +
                   "f_sharp_mean,g_mean,g_sharp_mean,a_mean,a_sharp_mean,b_mean," +
                   "c_std,c_sharp_std,d_std,d_sharp_std,e_std,f_std," +
                   "f_sharp_std,g_std,g_sharp_std,a_std,a_sharp_std,b_std," +
                   "mfcc_mean_0,mfcc_mean_1,mfcc_mean_2,mfcc_mean_3,mfcc_mean_4," +
                   "mfcc_mean_5,mfcc_mean_6,mfcc_mean_7,mfcc_mean_8,mfcc_mean_9," +
                   "mfcc_mean_10,mfcc_mean_11,mfcc_mean_12," +
                   "mfcc_std_0,mfcc_std_1,mfcc_std_2,mfcc_std_3,mfcc_std_4," +
                   "mfcc_std_5,mfcc_std_6,mfcc_std_7,mfcc_std_8,mfcc_std_9," +
                   "mfcc_std_10,mfcc_std_11,mfcc_std_12," +
                   "contr_mean_0,contr_mean_1,contr_mean_2,contr_mean_3," +
                   "contr_mean_4,contr_mean_5,contr_mean_6," +
                   "contr_std_0,contr_std_1,contr_std_2,contr_std_3," +
                   "contr_std_4,contr_std_5,contr_std_6," +
                   "cent_mean,cent_std,cent_skew," +
                   "row_mean,row_std,row_skew," +
                   "zrate_mean,zrate_std,zrate_skew\n"
                   )

for file in files:
    # # Load music and show its data # #
    # audio = librosa.util.example_audio_file()
    audio = file
    y, sr = librosa.load(audio)

    print('Audio Sampling Rate: ' + str(sr) + ' samples/sec')
    print('Total Samples: ' + str(np.size(y)))
    secs = np.size(y) / sr
    print('Audio Length: ' + str(secs)+' s')
    # IPython.display.Audio(audio)


    # # Feature Extraction # #
    y_harmonic, y_percussive = librosa.effects.hpss(y)
    # plt.figure(figsize=(15, 5))
    # librosa.display.waveplot(y_harmonic, sr=sr, alpha=0.25)
    # librosa.display.waveplot(y_percussive, sr=sr, color='r', alpha=0.5)
    # plt.title('Harmonic + Percussive')
    # plt.show()


    # -- Beat Extraction
    # = An estimate of the tempo (bpm)
    tempo, beat_frames = librosa.beat.beat_track(y=y_harmonic, sr=sr)
    print('Detected Tempo: ' + str(tempo) + ' beats/min')
    # beat_times = librosa.frames_to_time(beat_frames, sr=sr)
    # beat_time_diff = np.ediff1d(beat_times)
    # beat_nums = np.arange(1, np.size(beat_times))

    # fig, ax = plt.subplots()
    # fig.set_size_inches(15, 5)
    # ax.set_ylabel("Time difference (s)")
    # ax.set_xlabel("Beats")
    # g = sns.barplot(beat_nums, beat_time_diff, palette="BuGn_d", ax=ax)
    # g = g.set(xticklabels=[])
    # plt.show()

    # -- Chroma Energy Normalized (CENS)
    # =  12 equal-tempered pitch classes of western-type music
    chroma = librosa.feature.chroma_cens(y=y_harmonic, sr=sr)
    # plt.figure(figsize=(15, 5))
    # librosa.display.specshow(chroma, y_axis='chroma', x_axis='time')
    # plt.colorbar()
    # plt.show()

    # -- Calculate MFCCs
    # = Mel-frequency cepstral coefficients
    # = a nonlinear "spectrum-of-a-spectrum"
    # = the frequency bands are equally spaced on the mel scale
    #   which approximates the human auditory system's response
    mfccs = librosa.feature.mfcc(y=y_harmonic, sr=sr, n_mfcc=13)
    # plt.figure(figsize=(15, 5))
    # librosa.display.specshow(mfccs, x_axis='time')
    # plt.colorbar()
    # plt.title('MFCC')
    # plt.show()

    # -- Spectral Centroid
    # = indicate where the "center of the mass" of the spectrum is
    cent = librosa.feature.spectral_centroid(y=y, sr=sr)
    # plt.figure(figsize=(15, 5))
    # plt.subplot(1, 1, 1)
    # plt.semilogy(cent.T, label='Spectral centroid')
    # plt.ylabel('Hz')
    # plt.xticks([])
    # plt.xlim([0, cent.shape[-1]])
    # plt.legend()
    # plt.show()

    # -- Spectral Contrast
    # = represent the special characteristics of a music piece
    # = consider the spectral peak and valley in each sub-band separately
    #   Spectral peaks: correspond to harmonic components
    #   Spectral valley: reflect the the spectral contrast distribution
    # = Spectral peak - spectral valley
    contrast = librosa.feature.spectral_contrast(y=y_harmonic, sr=sr)
    # plt.figure(figsize=(15, 5))
    # librosa.display.specshow(contrast, x_axis='time')
    # plt.colorbar()
    # plt.ylabel('Frequency bands')
    # plt.title('Spectral contrast')
    # plt.show()

    # -- Spectral Rolloff
    # = Nth percentile of power spectral distribution
    # = distinguishing voiced speech from unvoiced
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
    # plt.figure(figsize=(15, 5))
    # plt.semilogy(rolloff.T, label='Roll-off frequency')
    # plt.ylabel('Hz')
    # plt.xticks([])
    # plt.xlim([0, rolloff.shape[-1]])
    # plt.legend()
    # plt.show()


    # -- Zero Crossing Rate
    # = a point in a digital audio file where the sample is at zero amplitude
    zrate = librosa.feature.zero_crossing_rate(y_harmonic)
    # plt.figure(figsize=(15, 5))
    # plt.semilogy(zrate.T, label='Fraction')
    # plt.ylabel('Fraction per Frame')
    # plt.xticks([])
    # plt.xlim([0, rolloff.shape[-1]])
    # plt.legend()
    # plt.show()


    # # Feature Generation # #
    # -- Chroma Energy Normalized
    # = contain 12 element representation of chroma energy throughout the song
    print("--- Chroma Energy Normalised ---")
    chroma_mean = np.mean(chroma, axis=1)
    chroma_std = np.std(chroma, axis=1)
    # plot the summary
    # octave = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    # plt.figure(figsize=(15, 5))
    # plt.title('Mean CENS')
    # sns.barplot(x=octave, y=chroma_mean)
    #
    # plt.figure(figsize=(15, 5))
    # plt.title('SD CENS')
    # sns.barplot(x=octave, y=chroma_std)
    # # Generate the chroma Dataframe
    # chroma_df = pd.DataFrame()
    # for i in range(0, 12):
    #     chroma_df['chroma_mean_' + str(i)] = chroma_mean[i]
    # for i in range(0, 12):
    #     chroma_df['chroma_std_' + str(i)] = chroma_mean[i]
    # chroma_df.loc[0] = np.concatenate((chroma_mean, chroma_std), axis=0)
    # print(chroma_df)
    # plt.show()

    # -- MFCCs
    print("------------- MFCCs -------------")
    mfccs_mean = np.mean(mfccs, axis=1)
    mfccs_std = np.std(mfccs, axis=1)

    # coeffs = np.arange(0, 13)
    # plt.figure(figsize=(15, 5))
    # plt.title('Mean MFCCs')
    # sns.barplot(x=coeffs, y=mfccs_mean)

    # plt.figure(figsize=(15, 5))
    # plt.title('SD MFCCs')
    # sns.barplot(x=coeffs, y=mfccs_std)
    # Generate the chroma Dataframe
    # mfccs_df = pd.DataFrame()
    # for i in range(0,13):
    #     mfccs_df['mfccs_mean_' + str(i)] = mfccs_mean[i]
    # for i in range(0,13):
    #     mfccs_df['mfccs_std_' + str(i)] = mfccs_mean[i]
    # mfccs_df.loc[0] = np.concatenate((mfccs_mean, mfccs_std), axis=0)
    # print(mfccs_df)

    # -- Spectral Features
    # > Spectral Centroid
    print("----- Spectral Centroid -----")
    cent_mean = np.mean(cent)
    cent_std = np.std(cent)
    cent_skew = scipy.stats.skew(cent, axis=1)[0]
    print('Mean: ' + str(cent_mean))
    print('SD: ' + str(cent_std))
    print('Skewness: ' + str(cent_skew))

    # > Spectral Contrast
    print("----- Spectral Contrast -----")
    contrast_mean = np.mean(contrast, axis=1)
    contrast_std = np.std(contrast, axis=1)

    # conts = np.arange(0, 7)
    # plt.figure(figsize=(15, 5))
    # plt.title('Mean Spectral Contrast')
    # sns.barplot(x=conts, y=contrast_mean)
    #
    # plt.figure(figsize=(15, 5))
    # plt.title('SD Spectral Contrast')
    # sns.barplot(x=conts, y=contrast_std)
    # Generate the chroma Dataframe
    contrast_df = pd.DataFrame()

    # > Spectral Rolloff
    print("------ Spectral Rolloff ------")
    rolloff_mean = np.mean(rolloff)
    rolloff_std = np.std(rolloff)
    rolloff_skew = scipy.stats.skew(rolloff, axis=1)[0]
    print('Mean: ' + str(rolloff_mean))
    print('SD: ' + str(rolloff_std))
    print('Skewness: ' + str(rolloff_skew))

    # spectral_df = pd.DataFrame()
    # collist = ['cent_mean', 'cent_std', 'cent_skew']
    # for i in range(0, 7):
    #     collist.append('contrast_mean_' + str(i))
    # for i in range(0, 7):
    #     collist.append('contrast_std_' + str(i))
    # collist = collist + ['rolloff_mean', 'rolloff_std', 'rolloff_skew']
    # for c in collist:
    #     spectral_df[c] = 0
    # data = np.concatenate(([cent_mean, cent_std, cent_skew],
    #                        contrast_mean, contrast_std,
    #                        [rolloff_mean, rolloff_std, rolloff_std]),
    #                       axis=0)
    # spectral_df.loc[0] = data
    # print(spectral_df)

    # -- Zero Crossing Rate
    print("---- Zero Crossing Rate ----")
    zrate_mean = np.mean(zrate)
    zrate_std = np.std(zrate)
    zrate_skew = scipy.stats.skew(zrate, axis=1)[0]
    # print('Mean: ' + str(zrate_mean))
    # print('SD: ' + str(zrate_std))
    # print('Skewness: ' + str(zrate_skew))

    # zrate_df = pd.DataFrame()
    # zrate_df['zrate_mean'] = 0
    # zrate_df['zrate_std'] = 0
    # zrate_df['zrate_skew'] = 0
    # zrate_df.loc[0] = [zrate_mean, zrate_std, zrate_skew]
    # zrate_df.loc[0] = [zrate_mean, zrate_std, zrate_skew]
    # print(zrate_df)

    # -- Beat and Tempo
    print("----- Beat and Tempo -----")
    # beat_df = pd.DataFrame()
    # beat_df['tempo'] = tempo
    # beat_df.loc[0] = tempo
    # print(beat_df)

    # # Generate the Final DataFrame # #
    # print("---------- Final ----------")
    # final_df = pd.concat((chroma_df, mfccs_df,
    #                       spectral_df, zrate_df, beat_df),
    #                      axis=1)
    # print(final_df.head())

    chm_str = str(list(chroma_mean)).replace('[', '').replace(']', ',')
    chs_str = str(list(chroma_std)).replace('[', '').replace(']', ',')
    mfm_str = str(list(mfccs_mean)).replace('[', '').replace(']', ',')
    mfs_str = str(list(mfccs_std)).replace('[', '').replace(']', ',')
    ctm_str = str(list(contrast_mean)).replace('[', '').replace(']', ',')
    cts_str = str(list(contrast_std)).replace('[', '').replace(']', ',')
    data = file.split('\\')[5] + "," + file.split('\\')[4]
    with open("music_ext.csv", "a") as txt:
        txt.writelines(file.split('\\')[5] + "," +
                       file.split('\\')[4] + "," +
                       "%f,%f," % (secs, np.size(y)) +
                       chm_str + chs_str + mfm_str + mfs_str + ctm_str + cts_str +
                       "%f,%f,%f,%f,%f,%f,%f,%f,%f\n"
                       % (cent_mean, cent_std, cent_skew,
                           rolloff_mean, rolloff_std, rolloff_skew,
                           zrate_mean, zrate_std, zrate_skew))



