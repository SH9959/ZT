
import pyaudio
from pydub.playback import play
from pydub import AudioSegment
# import sys

savepath = '/home/kuavo/catkin_dt/src/voice_pkg/temp_record/playnew.wav'

def playsound_work():
    play(AudioSegment.from_wav(savepath))
    
from playsound import playsound
def play_sound():
    playsound(savepath)


import sounddevice as sd
import soundfile as sf

def sd_play():
    filename = '/home/kuavo/catkin_dt/src/voice_pkg/temp_record/iamhere.mp3'
    data, fs = sf.read(filename)  # 读取音频文件

    # 指定设备索引或名称播放音频
    device_name = 'Bothlent UAC Dongle: USB Audio'  # 你可以根据实际情况替换这里的设备名称

    sd.play(data, fs, device=1)

def audio_play():
    import pyaudio
import wave

def play_audio():
    # 打开WAV音频文件
    wf = wave.open(savepath, 'rb')

    # 创建PyAudio对象
    p = pyaudio.PyAudio()

    # 打开输出流
    print("channel:", wf.getnchannels())
    stream = p.open(format=p.get_format_from_width(wf.getsampwidth()),
                    channels=1,
                    rate=wf.getframerate(),
                    output=True,
                    output_device_index=2
                    )

    # 读取数据
    data = wf.readframes(4000)

    # 播放
    while data:
        stream.write(data)
        data = wf.readframes(4000)

    # 停止和关闭流
    stream.stop_stream()
    stream.close()

    # 关闭PyAudio
    p.terminate()



if __name__ == '__main__':
    # playsound_work()
    play_sound()
    # sd_play()
    # play_audio()
