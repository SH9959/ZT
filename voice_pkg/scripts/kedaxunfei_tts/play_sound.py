
from pydub.playback import play
from pydub import AudioSegment
# import sys

savepath = '/home/kuavo/catkin_dt/src/voice_pkg/temp_record/play.mp3'
savepath = '/home/kuavo/Music/001.mp3'

def playsound_work():
    play(AudioSegment.from_mp3(savepath))
    
from playsound import playsound
def play_sound():
    playsound(savepath)
if __name__ == '__main__':
    playsound_work()