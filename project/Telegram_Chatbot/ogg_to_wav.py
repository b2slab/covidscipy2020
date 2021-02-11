import os,sys
from pydub import AudioSegment

orig_song = sys.argv[1]
dest_song = os.path.splitext(sys.argv[1])[0]+'.wav'

def convert_ogg_to_wav():
    song = AudioSegment.from_ogg(orig_song)
    song.export(dest_song, format="wav")
    
if __name__ == '__main__':
    convert_ogg_to_wav()
