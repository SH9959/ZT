#!/home/lemon/anaconda3/envs/zt/bin/python
# -*- coding:utf-8 -*-
#
#   author: iflytek
#
#  本demo测试时运行的环境为：Windows + Python3.7
#  本demo测试成功运行时所安装的第三方库及其版本如下，您可自行逐一或者复制到一个新的txt文件利用pip一次性安装：
#   cffi==1.12.3
#   gevent==1.4.0
#   greenlet==0.4.15
#   pycparser==2.19
#   six==1.12.0
#   websocket==0.2.1
#   websocket-client==0.56.0
#
#  语音听写流式 WebAPI 接口调用示例 接口文档（必看）：https://doc.xfyun.cn/rest_api/语音听写（流式版）.html
#  webapi 听写服务参考帖子（必看）：http://bbs.xfyun.cn/forum.php?mod=viewthread&tid=38947&extra=
#  语音听写流式WebAPI 服务，热词使用方式：登陆开放平台https://www.xfyun.cn/后，找到控制台--我的应用---语音听写（流式）---服务管理--个性化热词，
#  设置热词
#  注意：热词只能在识别的时候会增加热词的识别权重，需要注意的是增加相应词条的识别率，但并不是绝对的，具体效果以您测试为准。
#  语音听写流式WebAPI 服务，方言试用方法：登陆开放平台https://www.xfyun.cn/后，找到控制台--我的应用---语音听写（流式）---服务管理--识别语种列表
#  可添加语种或方言，添加后会显示该方言的参数值
#  错误码链接：https://www.xfyun.cn/document/error-code （code返回错误码时必看）
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
import audioop
import os
import wave
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
import speech_recognition as sr 
# pip install SpeechRecognition

from pydub.playback import play
from pydub import AudioSegment


import rospy
# from voice_pkg.srv import VoiceSrv, VoiceSrvResponse

STATUS_FIRST_FRAME = 0  # 第一帧的标识
STATUS_CONTINUE_FRAME = 1  # 中间帧标识
STATUS_LAST_FRAME = 2  # 最后一帧的标识

savepath = '/home/kuavo/catkin_dt/src/voice_pkg/temp_record/mic.wav'


class Ws_Param(object):
    # 初始化
    def __init__(self, APPID, APIKey, APISecret, AudioFile):
        self.APPID = APPID
        self.APIKey = APIKey
        self.APISecret = APISecret
        self.AudioFile = AudioFile
        self.result = ''
        self.flagover = False  # 提示音是否已经读完

        # 公共参数(common)
        self.CommonArgs = {"app_id": self.APPID}
        # 业务参数(business)，更多个性化参数可在官网查看
        self.BusinessArgs = {"domain": "iat", "language": "zh_cn", "accent": "mandarin", "vinfo":0,"vad_eos":10000,"dwa":"wpgs"}

    # 生成url
    def create_url(self):
        url = 'wss://ws-api.xfyun.cn/v2/iat'
        # 生成RFC1123格式的时间戳
        now = datetime.now()
        date = format_date_time(mktime(now.timetuple()))

        # 拼接字符串
        signature_origin = "host: " + "ws-api.xfyun.cn" + "\n"
        signature_origin += "date: " + date + "\n"
        signature_origin += "GET " + "/v2/iat " + "HTTP/1.1"
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
        return url


def iat_web_api(req, iter=1):
    # input = req.input
    input = req

    # 打开麦克风录音
    r = sr.Recognizer()
    r.pause_threshold = 0.8   # minimum length of silence (in seconds) that will register as the end of a phrase.  
    with sr.Microphone() as source1:
        # calibrate ambient noise
        r.adjust_for_ambient_noise(source1, duration=0.8)
    print('now ambient noise energy = ', r.energy_threshold)
    for i in range(iter):
        try:
            print('hihi')
            if input == 'zai':
                run_iamhere()
            thread.start_new_thread(run_ding, ())
            while wsParam.flagover == False: continue
            with sr.Microphone() as source:
                time1 = datetime.now()
                print("I start listen....", time1)
                audio = r.listen(source, phrase_time_limit=8)
                print("I finish listen....", datetime.now()-time1)

                # get wav data from AudioData object 
                wav = audio.get_wav_data(convert_rate=16000, convert_width=2) # width=2 gives 16bit audio.

                # 保存 wav
                audio_file = wsParam.AudioFile
                wf = wave.open(audio_file, 'wb')
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(16000)
                wf.writeframes(wav)
                wf.close()
                print('sr is over', datetime.now())

        except sr.exceptions.WaitTimeoutError as e:
            print('超时重来')
            continue  
    # 收到websocket消息的处理
    def on_message(ws, message):
        try:
            code = json.loads(message)["code"]
            sid = json.loads(message)["sid"]
            if code != 0:
                errMsg = json.loads(message)["message"]
                print("sid:%s call error:%s code is:%s" % (sid, errMsg, code))
            else:
                data = json.loads(message)["data"]["result"]["ws"]
                iflast = json.loads(message)["data"]["result"]["ls"]
                pgs = json.loads(message)["data"]["result"]["pgs"]
                result = ""
                for i in data:
                    for w in i["cw"]:
                        result += w["w"]
                if pgs == 'rpl':
                    wsParam.result = result
                else:
                    wsParam.result += result
                print(wsParam.result)
                if iflast:
                    ws.close()
        except Exception as e:
            print("receive msg,but parse exception:", e, datetime.now())
            ws.close()
            return

    # 收到websocket错误的处理
    def on_error(ws, error):
        print("### error:", error, datetime.now())
        ws.close()

    # 收到websocket关闭的处理
    def on_close(ws,a,b):
        print("### closed ###", datetime.now())
        print(a)
        print(b)

    # 收到websocket连接建立的处理
    def on_open(ws):
        def run_iamhere():
          finename = '/home/kuavo/catkin_dt/src/voice_pkg/temp_record/iamhere.mp3'
          play(AudioSegment.from_mp3(finename)+10)

        def run_ding():
          finename = '/home/kuavo/catkin_dt/src/voice_pkg/temp_record/ding_cut.mp3'
          play(AudioSegment.from_mp3(finename)+20)
          wsParam.flagover = True
          
        def run(*args):
            frameSize = 3000  # 每一帧的音频大小
            intervel = 0.04  # 发送音频间隔(单位:s)
            status = STATUS_FIRST_FRAME  # 音频的状态信息，标识音频是第一帧，还是中间帧、最后一帧           
            
            # 获取音频之后需要调用api识别
            with open(wsParam.AudioFile, "rb") as fp:
                print('start audio')
                while True:
                    buf = fp.read(frameSize)
                    # 文件结束
                    if not buf:
                        status = STATUS_LAST_FRAME
                    # 第一帧处理
                    # 发送第一帧音频，带business 参数
                    # appid 必须带上，只需第一帧发送
                    if status == STATUS_FIRST_FRAME:
                        d = {"common": wsParam.CommonArgs,
                                "business": wsParam.BusinessArgs,
                                "data": {"status": 0, "format": "audio/L16;rate=16000",
                                        "audio": str(base64.b64encode(buf), 'utf-8'),
                                        "encoding": "raw"}}
                        d = json.dumps(d)
                        print('send first audio')
                        ws.send(d)
                        status = STATUS_CONTINUE_FRAME
                    # 中间帧处理
                    elif status == STATUS_CONTINUE_FRAME:
                        d = {"data": {"status": 1, "format": "audio/L16;rate=16000",
                                        "audio": str(base64.b64encode(buf), 'utf-8'),
                                        "encoding": "raw"}}
                        print('send mid audio')
                        ws.send(json.dumps(d))
                    # 最后一帧处理
                    elif status == STATUS_LAST_FRAME:
                        d = {"data": {"status": 2, "format": "audio/L16;rate=16000",
                                        "audio": str(base64.b64encode(buf), 'utf-8'),
                                        "encoding": "raw"}}
                        print('send last audio')
                        ws.send(json.dumps(d))
                        time.sleep(1)
                        break
                    # 模拟音频采样间隔
                    time.sleep(intervel)
                ws.close()
        time1 = datetime.now()
        thread.start_new_thread(run, ())
        time2 = datetime.now()
        print('speechrecognition cost = ', time2 - time1)


    with open('/home/kuavo/catkin_dt/config_dt.json', 'r') as fj:
        config = json.load(fj)
    APPID, APISecret, APIKey = config['kedaxunfei_appid'], config['kedaxunfei_apiSecret'], config['kedaxunfei_appkey']
    wsParam = Ws_Param(APPID=APPID, APISecret=APISecret,
                       APIKey=APIKey,
                       AudioFile=savepath)
    websocket.enableTrace(False)
    wsUrl = wsParam.create_url()
    ws = websocket.WebSocketApp(wsUrl, on_message=on_message, on_error=on_error, on_close=on_close)
    ws.on_open = on_open
    ws.run_forever(sslopt={"cert_reqs": ssl.CERT_NONE})
    print(f"socket time cost final:{datetime.now()}")
    
    print(wsParam.result)
    print('-------iat is over----------')
    # return VoiceSrvResponse(wsParam.result)
    return wsParam.result
    

def kedaxunfei_iat_server():
    rospy.init_node('kedaxunfei_iat_server')
    s = rospy.Service('kedaxunfei_iat', VoiceSrv, iat_web_api)
    print("ready to kedaxunfei iat")
    rospy.spin()

if __name__ == "__main__":
    # kedaxunfei_iat_server()
    iat_web_api('ding', iter=3)
