import os

from pydub import AudioSegment
import pandas as pd


class AudioImport:
    def __init__(self, audios_dir):
        self.audios_dir = audios_dir

    def cut_audio(self, audio_file):
        audio_file_path = os.path.join(self.audios_dir, "converted", audio_file)
        t1 = 0  # Works in milliseconds
        t2 = 4000
        audio = AudioSegment.from_wav(audio_file_path)
        audio = audio[t1:t2]
        audio.export(os.path.join(self.audios_dir, "cuts", audio_file), format="wav") #Exports to a wav file in the current path

    def convert_to_wav(self, file):
        new_name = file.replace(" ", "_")
        os.rename(os.path.join(self.audios_dir, file), os.path.join(self.audios_dir, new_name))
        filename = "".join(new_name.split('.')[:-1])
        os.system(f'ffmpeg -i {os.path.join(self.audios_dir, new_name)} {os.path.join(self.audios_dir, "converted", filename)}.wav')

    @staticmethod
    def _is_not_format(file, format):
        return "." in file and not file.endswith(format)

    @staticmethod
    def _is_format(file, format):
        return "." in file and file.endswith(format)

    def prepare_audios(self):
        output_dir = os.path.join(self.audios_dir, "cuts")
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        output_dir = os.path.join(self.audios_dir, "converted")
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        print("Converting files to wav...")
        [self.convert_to_wav(file) for file in os.listdir(self.audios_dir) if self._is_not_format(file, 'wav')]
        print("Cutting files to 4s...")
        [self.cut_audio(file) for file in os.listdir(os.path.join(self.audios_dir, "converted"))]

    def move_files(self):
        print("Moving files...")
        db_audios_dir = os.path.join("database", "audio")
        output_dir = os.path.join(db_audios_dir, "fold13")
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        current_dir = os.path.join(self.audios_dir, "converted")
        files = [file for file in os.listdir(current_dir) if self._is_format(file, 'wav')]
        [os.rename(os.path.join(current_dir, file), os.path.join(output_dir, file)) for file in files]

    def add_to_metadata(self):
        print("Adding to metadata... ")
        files = os.listdir(os.path.join("database", "audio", "fold13"))
        rows = [[file, "-", "-", "-", "-", 13, 10, "cough"] for file in files]
        metadata = pd.read_csv(os.path.join('database', 'metadata', 'UrbanSound8K.csv'), index_col=0)
        new_df = pd.DataFrame(rows, columns=metadata.columns)
        merged_df = metadata.append(new_df)
        merged_df.to_csv(os.path.join('database', 'metadata', 'UrbanSound8K.csv'), index=False)

if __name__ == '__main__':
    importer = AudioImport("tos")
    # importer.prepare_audios()
    importer.move_files()
    importer.add_to_metadata()
