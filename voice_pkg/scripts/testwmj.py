import time
import _thread as thread
import subprocess


class Test():
    def __init__(self,):
        self.lastproc = None
        self.lastplayproc = None
        
    def text2speech(self, input='', index=0):
        global interrupt
        # with open("/home/kuavo/catkin_dt/src/voice_pkg/temp_record/tts_sentence.txt", "w") as f:
        #   f.write(input)
        ttsproc = subprocess.Popen(["python3", "/home/kuavo/catkin_dt/src/voice_pkg/scripts/kedaxunfei_tts/tts_as_service.py", input])
        while ttsproc.poll() is None:
          print('tts process is working')
          if interrupt:
            ttsproc.kill()
            return
          time.sleep(1)
        
        while self.lastplayproc and self.lastplayproc.poll() is None:
          print('last process is working, waiting')
          if interrupt:
            self.lastplayproc.kill()
            return
          time.sleep(1)

        playproc = subprocess.Popen(["python3", "/home/kuavo/catkin_dt/src/voice_pkg/scripts/kedaxunfei_tts/playsound.py"])
        if index == 1000: 
          # 同步播放
          while playproc.poll() is None:
              print('play process is working')
              if interrupt:
                playproc.kill()
                return
              time.sleep(1)
          self.lastplayproc = None
        else:
          # 异步播放:
          self.lastplayproc = playproc
          # 等待的时间必不可少，因为会有playsound和tts的读写同一个文件的冲突，因此先playsound再让tts访问 play.wav
          time.sleep(0.15)
        return 'tts is over'

def change_int():
    global interrupt
    time.sleep(6)
    interrupt = True
    print('----------')

if __name__ == '__main__':
    a = Test()
    interrupt = False
    thread.start_new_thread(change_int, ())
    for i in ['获取下一个目标点1111', '获取下一个目标点2', '获取下一个目标点3']:
        a.text2speech(i, 0)
        if interrupt:
          break
