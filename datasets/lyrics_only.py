import torch
from math import ceil, floor
from torch.utils.data import Dataset, DataLoader
from mido import MidiFile, tick2second
import os
from scipy import fft
from scipy.fft import dct
import soundfile as sf
# import matplotlib.pyplot as plt
import numpy as np
from torchvision.utils import save_image



OGG_EXT = ".ogg"


SAMPLE_RATE = 48000
NUM_SEGMENTS = 2 ** 9
INT16_MAX = 2**15
AUDIO_DURATION_IN_BYTES = 2**18
AUDIO_DURATION = 5.4613333   # assuming SAMPLERATE=48000 we have AUDIO_DURATION * SAMPLERATE=2**18

class LyricsOnly(Dataset):
    def __init__(self, ds_path, audio_segment_dir=None, audio_duration=AUDIO_DURATION, ignore_empty=True, do_preprocessing=True, save_songs=False, save_pre_fourier=False, save_fourier=True, rgb_grouping=True, save_collapsed=False, save_fourier_image=False):
        self.ds_path = ds_path
        self.audio_duration = audio_duration
        self.ignore_empty = ignore_empty
        self.save_songs = save_songs
        self.save_pre_fourier = save_pre_fourier
        self.save_fourier = save_fourier
        self.rgb_grouping = rgb_grouping
        self.save_collapsed = save_collapsed
        self.save_fourier_image = save_fourier_image
        if audio_segment_dir is None:
            audio_segment_dir = os.path.join(ds_path, '..', 'audio_segments')
        self.audio_segment_dir = audio_segment_dir
        if rgb_grouping:
            self.track_to_channel = {'vocals': 0,
                                     'song': 1, 'guitar': 1, 'rhythm': 1,
                                     'drums_1': 2, 'drums_2': 2, 'drums_3': 2, 'drums_4': 2,
                                     }
        else:
            self.track_to_channel = {'vocals': 0, 'song': 1, 'guitar': 2, 'rhythm': 3,
                                     'drums_1': 4, 'drums_2': 5, 'drums_3': 6, 'drums_4': 7}
        if do_preprocessing:
            self._split_audio_to_segments()

    def __getitem__(self, item: int):
        processed_segment_dir = os.path.join(self.audio_segment_dir, f'{item}')
        fourier = np.load(os.path.join(processed_segment_dir, 'fourier.npy'))
        with open(os.path.join(processed_segment_dir, 'lyrics.txt'), 'r') as f:
            lyrics = f.read()
        return {'image': fourier, 'text': lyrics}

    def __len__(self):
        return len(os.listdir(self.audio_segment_dir))

    def identify_channel(self, filename):
        channel_type = filename[:-len(OGG_EXT)]
        return self.track_to_channel[channel_type]

    def _split_audio_to_segments(self):
        index = 0
        for dir in os.listdir(self.ds_path):
            dir_path = os.path.join(self.ds_path, dir)
            if os.path.isdir(dir_path):

                midi_file_path = os.path.join(dir_path, 'notes.mid')

                if not os.path.exists(midi_file_path):
                    continue

                print('working on song: {}'.format(dir))
                try:

                    t2lyrics = self.cut_lyrics_to_parts(midi_file_path, self.audio_duration)
                    pre_fourier_per_section = dict()
                    fourier_per_section = dict()

                    # opening other audio files and cutting them to parts according to lyrics
                    for audio_file in os.listdir(dir_path):

                        if audio_file.endswith('.ogg') and audio_file[:-4] in self.track_to_channel:

                            audio_file_path = os.path.join(dir_path, audio_file)
                            audio, samplerate = sf.read(audio_file_path, dtype = np.int16)

                            for i, ((t_start, t_end), lyrics) in enumerate(t2lyrics.items()):

                                # print(f'section {i}: {lyrics}, {len(lyrics)}')
                                if len(lyrics) <= 3 and self.ignore_empty:
                                    continue

                                # making sure all directories exist
                                if not os.path.isdir(self.audio_segment_dir):
                                    os.makedirs(self.audio_segment_dir)
                                processed_song_dir = os.path.join(self.audio_segment_dir, dir)

                                section_start = int(t_start * samplerate)
                                section_end = section_start + ceil(self.audio_duration*samplerate)

                                audio_section = audio[section_start:section_end]
                                collapsed_audio_section = np.mean(audio_section, axis=1, dtype = np.int16)  # collapsing L/R to mono

                                if self.save_songs:
                                    sf.write(os.path.join(processed_section_dir, audio_file), collapsed_audio_section, samplerate)

                                if i not in pre_fourier_per_section:
                                    pre_fourier_per_section[i] = np.zeros((len(set(self.track_to_channel.values())), len(collapsed_audio_section)))
                                pre_fourier_per_section[i][self.identify_channel(audio_file)] = collapsed_audio_section

                    for i, (_, lyrics) in enumerate(t2lyrics.items()):
                        if i not in pre_fourier_per_section:
                            continue


                        waveform = pre_fourier_per_section[i]
                        instrument_number, waveform_length = waveform.shape
                        assert waveform_length == AUDIO_DURATION_IN_BYTES

                        segmented_waveform = waveform.reshape(instrument_number, NUM_SEGMENTS, NUM_SEGMENTS)
                        cosine_transform = dct(segmented_waveform)


                        fourier_per_section[i] = cosine_transform

                        if self.save_pre_fourier:
                            processed_song_dir = os.path.join(self.audio_segment_dir, dir)
                            processed_section_dir = os.path.join(processed_song_dir, f'section_{i:>02}_{lyrics.strip()}')
                            np.save(os.path.join(processed_section_dir, f'pre_fourier.npy'), waveform)
                            if self.save_collapsed:
                                sf.write(os.path.join(processed_section_dir, 'collapsed.ogg'), waveform.transpose(), samplerate)

                        if self.save_fourier:
                            processed_segment_dir = os.path.join(self.audio_segment_dir, f'{index}')
                            if not os.path.isdir(processed_segment_dir):
                                os.makedirs(processed_segment_dir)
                            index += 1
                            np.save(os.path.join(processed_segment_dir, f'fourier.npy'), cosine_transform)
                            if self.save_fourier_image:
                                save_image(torch.tensor(cosine_transform), os.path.join(processed_segment_dir, f'fourier.png'))
                            with open(os.path.join(processed_segment_dir,'lyrics.txt'), 'w') as f:
                                f.write(f'{lyrics.strip()}')
                except:
                    print(f'failed to process {dir_path}')
                    continue



    @staticmethod
    def cut_lyrics_to_parts(midi_file_path, audio_duration):
        """get lyrics from midi file, and cut it to audio_duration long parts
        :return dict: { (t_start, t_end): lyrics }"""
        t2lyrics = dict()
        # print(midi_file_path)
        mid = MidiFile(midi_file_path, clip=True)
        lyrics_track = [t for t in mid.tracks if t.name == 'PART VOCALS'][0]
        tempo_track = mid.tracks[0]
        assert tempo_track[0].type == 'set_tempo'

        i = 0
        j = 0
        current_beat = 0
        next_change = 0
        current_time = 0
        total_time = 0
        cur_lyrics = ''
        start_time = 0

        while i < len(lyrics_track):
            if next_change <= current_beat:
                #changing tempo
                if tempo_track[j].type != 'set_tempo':
                    j += 1
                    continue
                tempo = tempo_track[j].tempo
                j += 1
                next_change += tempo_track[j].time

            if lyrics_track[i].type == 'lyrics':
                cur_lyrics += lyrics_track[i].text
                cur_lyrics += ' '

            current_beat += lyrics_track[i].time
            current_time += tick2second(lyrics_track[i].time, mid.ticks_per_beat, tempo)
            total_time += tick2second(lyrics_track[i].time, mid.ticks_per_beat, tempo)

            if current_time >= audio_duration:
                cur_lyrics = cur_lyrics.replace('- ', '').replace('+ ', '').replace(' +', '').replace(' -', '').replace('#', '').replace('=', '')
                # TODO: fix this next line to give the actual t_start and t_end. currently its t_start + audio_duration so some words are cut in the middle.
                t2lyrics[(start_time, start_time + audio_duration)] = cur_lyrics
                cur_lyrics = ''
                current_time = 0
                start_time = total_time
            i += 1
        return t2lyrics

    @staticmethod
    def waveform_from_fourier(fourier):
        """
        Converts the Fourier representation back to mono waveform
        """
        assert fourier.shape == (NUM_SEGMENTS, NUM_SEGMENTS)

        return np.array(np.reshape(fft.idct((fourier)), [-1]), np.int16)

def lower_samplerate(song_path='audio_segments/Foo Fighters - Everlong/section_11_der When I sing along with you /vocals.ogg',
                     new_samplerate=6000):
    audio, samplerate = sf.read(song_path)
    print(f'audio: {len(audio)}, samplerate: {samplerate}')
    downsampled_audio = [amplitude for i, amplitude in enumerate(audio) if i % (samplerate // new_samplerate) == 0]
    print(f'downsampled_audio: {len(downsampled_audio)}, new_samplerate: {new_samplerate}')
    sf.write('tmp_song.ogg', downsampled_audio, new_samplerate)
    print('done')

def main():
    print('starting')
    ds_path = os.path.join('songs')
    ds = LyricsOnly(ds_path, do_preprocessing=False)
    print(f'len(ds): {len(ds)}')
    dl = DataLoader(ds, batch_size=4, shuffle=True, num_workers=0)

    for i, sample in enumerate(dl):
        print(f'batch {i}')
        print(sample)
        if i > 3:
            break

    sample = ds[0]
    print(sample["text"])
    # plt.show(sample["image"])
    fourier = torch.tensor(sample["image"])
    print(fourier.shape)
    print(fourier.max())
    print(fourier.min())
    print(fourier)
    fourier = fourier.numpy()
    for i in range(fourier.shape[0]):
        print(fourier[i].shape == (NUM_SEGMENTS, NUM_SEGMENTS))
        sound = ds.waveform_from_fourier(fourier[i])
        print(sound.shape)
        sf.write(f'tmp_song_{i}.ogg', sound, SAMPLE_RATE)

    print('done')


if __name__ == '__main__':
    main()