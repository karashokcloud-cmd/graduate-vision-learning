import sys, os
sys.path.insert(0, r'D:\ChatGPT\Chats\录音转写\runtime')
os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING']='1'
import av
_av_open=av.open
def compat_open(*args, **kwargs):
 kwargs.pop('metadata_errors',None)
 return _av_open(*args, **kwargs)
av.open=compat_open
from faster_whisper import WhisperModel
root=r'D:\ChatGPT\Chats\录音转写'
model=WhisperModel('small',device='cpu',compute_type='int8',download_root=os.path.join(root,'models'),cpu_threads=8)
segments,info=model.transcribe(r'C:\Users\28622\Downloads\弘毅大道.m4a',language='zh',beam_size=8,condition_on_previous_text=False,initial_prompt='高铁漆膜缺陷检测，飞机缺陷检测，数据集，二维检测，模型迁移，汇报，师姐。')
print('duration',info.duration,flush=True)
with open(os.path.join(root,'弘毅大道_复核.txt'),'w',encoding='utf-8') as f:
 for s in segments:
  line=f'[{int(s.start//60):02d}:{int(s.start%60):02d}–{int(s.end//60):02d}:{int(s.end%60):02d}] {s.text.strip()}'
  print(line,flush=True);f.write(line+'\n');f.flush()


