import subprocess
import websocket
import datetime
import hashlib
import base64
import hmac
import json
from urllib.parse import urlencode
import time
import ssl
from wsgiref.handlers import format_date_time
from datetime import datetime
from time import mktime
import _thread as thread
import os
import sys

from pydub.playback import play
from pydub import AudioSegment

import rospy
# from voice_pkg.srv import VoiceSrv, VoiceSrvResponse

STATUS_FIRST_FRAME = 0  # 第一帧的标识
STATUS_CONTINUE_FRAME = 1  # 中间帧标识
STATUS_LAST_FRAME = 2  # 最后一帧的标识

playstate = 'stop'
savepath = '/home/kuavo/catkin_dt/src/voice_pkg/temp_record/play.wav'

class Ws_Param(object):
    # 初始化
    def __init__(self, APPID, APIKey, APISecret, Text):
        self.APPID = APPID
        self.APIKey = APIKey
        self.APISecret = APISecret
        self.Text = Text

        # 公共参数(common)
        self.CommonArgs = {"app_id": self.APPID}
        # 业务参数(business)，更多个性化参数可在官网查看
        self.BusinessArgs = {"aue":"lame", "sfl":1, "auf":"audio/L16;rate=16000", "vcn":"x4_qige", "bgs":0, "tte":"utf8","speed":55}
        self.Data = {"status": 2, "text": str(base64.b64encode(self.Text.encode('utf-8')), "UTF8")}
        #使用小语种须使用以下方式，此处的unicode指的是 utf16小端的编码方式，即"UTF-16LE"”
        #self.Data = {"status": 2, "text": str(base64.b64encode(self.Text.encode('utf-16')), "UTF8")}

    # 生成url
    def create_url(self):
        url = 'wss://tts-api.xfyun.cn/v2/tts'
        # 生成RFC1123格式的时间戳
        now = datetime.now()
        date = format_date_time(mktime(now.timetuple()))

        # 拼接字符串
        signature_origin = "host: " + "ws-api.xfyun.cn" + "\n"
        signature_origin += "date: " + date + "\n"
        signature_origin += "GET " + "/v2/tts " + "HTTP/1.1"
        # 进行hmac-sha256进行加密
        signature_sha = hmac.new(self.APISecret.encode('utf-8'), signature_origin.encode('utf-8'),
                                 digestmod=hashlib.sha256).digest()
        signature_sha = base64.b64encode(signature_sha).decode(encoding='utf-8')

        authorization_origin = "api_key=\"%s\", algorithm=\"%s\", headers=\"%s\", signature=\"%s\"" % (
            self.APIKey, "hmac-sha256", "host date request-line", signature_sha)
        authorization = base64.b64encode(authorization_origin.encode('utf-8')).decode(encoding='utf-8')
        # 将请求的鉴权参数组合为字典
        v = {
            "authorization": authorization,
            "date": date,
            "host": "ws-api.xfyun.cn"
        }
        # 拼接鉴权参数，生成url
        url = url + '?' + urlencode(v)
        # print("date: ",date)
        # print("v: ",v)
        # 此处打印出建立连接时候的url,参考本demo的时候可取消上方打印的注释，比对相同参数时生成的url与自己代码生成的url是否一致
        # print('websocket url :', url)
        return url

def get_tts(textinput):
    def on_message(ws, message):
        try:
            message =json.loads(message)
            if message is None:
                print('发现 NULL 帧 返回！！')
                return
            code = message["code"]
            sid = message["sid"]
            data = message["data"]
            if data is None:
                print('发现空的！！！！')
                return
            audio = message["data"]["audio"]
            audio = base64.b64decode(audio)
            status = message["data"]["status"]
            # print(message)
            if status == 2:
                print("ws is closed")
                ws.close()
            if code != 0:
                errMsg = message["message"]
                print("sid:%s call error:%s code is:%s" % (sid, errMsg, code))
            else:
                with open(savepath, 'ab') as f:
                    print('write to file')
                    f.write(audio)

        except Exception as e:
            print("receive msg,but parse exception:", e, datetime.now())
            ws.close()


    # 收到websocket错误的处理
    def on_error(ws, error):
        print("### error:", error, datetime.now())


    # 收到websocket关闭的处理
    def on_close(ws,a,b):
        print("### closed ###", datetime.now())


    # 收到websocket连接建立的处理
    def on_open(ws):
        def run(*args):
            d = {"common": wsParam.CommonArgs,
                "business": wsParam.BusinessArgs,
                "data": wsParam.Data,
                }
            d = json.dumps(d)
            ws.send(d)
            if os.path.exists(savepath):
                os.remove(savepath)

        thread.start_new_thread(run, ())

    
    # 测试时候在此处正确填写相关信息即可运行
    wsParam = Ws_Param(APPID='c57ccaf5', APISecret='NjM0NjcxNmI4OGVhMWUzOTNhMDAxOTYx',
                       APIKey='b1d7d520b0c50e9442d0be07545b76d5',
                       Text=textinput)
    websocket.enableTrace(False)
    wsUrl = wsParam.create_url()
    ws = websocket.WebSocketApp(wsUrl, on_message=on_message, on_error=on_error, on_close=on_close)
    ws.on_open = on_open
    # ws.run_forever(sslopt={"cert_reqs": ssl.CERT_NONE}, ping_interval=2, ping_timeout=1)
    ws.run_forever(sslopt={"cert_reqs": ssl.CERT_NONE})
    print('-------tts is over----------')


# def tts_playsound(request):
#     input = request.input
#     index = request.index
def tts_playsound(input, index=1000):
    def playsound_work():
        global playstate
        playstate = 'playing'
        play(AudioSegment.from_mp3(savepath)+10)
        playstate = 'stop'

    print('text = ', input)
    get_tts(input)

    global playstate
    global lastproc
    print('playstate = ', playstate)
    # 同步播放
    if index == 1000:
        while True:
            if playstate == 'stop':
                playsound_work()
                # 等待的时间必不可少，因为会有playsound和tts的读写同一个文件的冲突，因此先playsound再让tts访问 play.wav
                time.sleep(0.15)
                return
    # 异步播放:
    # if lastproc:
    #     lastproc.wait()
    # lastproc = subprocess.Popen(["python3", "/home/kuavo/catkin_dt/src/voice_pkg/scripts/playsound.py"])
    while 1:
        if playstate == 'stop':
            thread.start_new_thread(playsound_work, ())

            # 等待的时间必不可少，因为会有playsound和tts的读写同一个文件的冲突，因此先playsound再让tts访问 play.wav
            time.sleep(0.15)

            break

    # 播放是否成功 需要检测喇叭
    # return VoiceSrvResponse('tts is over')
    return 'tts is over'

lastproc = None

def tts_playsound_terminate(input):
    print('text = ', input)
    get_tts(input)
    
    global lastproc
    if lastproc:
        lastproc.wait()

    # 异步播放:
    lastproc = subprocess.Popen(["python3", "/home/kuavo/catkin_dt/src/voice_pkg/scripts/playsound.py"])
    # 等待的时间必不可少，因为会有playsound和tts的读写同一个文件的冲突，因此先playsound再让tts访问 play.wav
    time.sleep(0.15)
    return lastproc


def kedaxunfei_tts_server():
    rospy.init_node('kedaxunfei_tts_server')
    s = rospy.Service('kedaxunfei_tts', VoiceSrv, tts_playsound)
    print("ready kedaxunfei tts")
    rospy.spin()

if __name__ == "__main__":
    # with open("/home/kuavo/catkin_dt/src/voice_pkg/temp_record/tts_sentence.txt", "r") as f:
    #   textinput = f.read()
    textinput = '你好啊，我叫七哥'
    get_tts(textinput)
    # # kedaxunfei_tts_server()
    # # tts_playsound('你好', index=0)
    # # tts_playsound('你不好，你好，你好你好', index=1000)
    # proc = tts_playsound_terminate('你好，你好！你好，你好！', lastproc=None, index=0)
    # # proc.kill()
    # thread.start_new_thread(tts_playsound_terminate, ('再见，再见，再见，再见！', proc, 0,))
    # print('yes')
    # time.sleep(1)
    # proc.kill()
    # # proc = tts_playsound_terminate('你好，你好，你好，你好！', proc, index=0)

