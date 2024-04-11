import subprocess
import time, os
# text = "你好啊，我叫基哥"
# inta = False
# ttsproc = subprocess.Popen(["python3", "/home/kuavo/catkin_dt/src/voice_pkg/scripts/kedaxunfei_tts/tts_as_service.py", text])
# while ttsproc.poll() is None:
#     print('tts process is working, interrupted:', inta)
#     if inta:
#         ttsproc.kill()
#         break
#     time.sleep(0.5)
cmd = " ".join(["aplay", "-D", "plughw:2,0", "/home/kuavo/catkin_dt/src/voice_pkg/temp_record/playnew.wav"])
exit_status = os.system(cmd)
print("-=-=-=-")