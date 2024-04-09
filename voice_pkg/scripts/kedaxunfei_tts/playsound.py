
from pydub.playback import play
from pydub import AudioSegment
# import sys

savepath = '/home/kuavo/catkin_dt/src/voice_pkg/temp_record/play.wav'
def playsound_work():
    play(AudioSegment.from_mp3(savepath)+10)
    
if __name__ == '__main__':
    playsound_work()