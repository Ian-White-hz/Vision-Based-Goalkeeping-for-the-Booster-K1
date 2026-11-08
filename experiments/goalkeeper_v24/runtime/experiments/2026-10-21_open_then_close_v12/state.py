import json
from pathlib import Path
import time
from bundle_support import BUNDLE_ROOT
HERE=BUNDLE_ROOT/'outputs'
HERE.mkdir(exist_ok=True)

def status(**updates):
    path=HERE/'status.json'
    old=json.loads(path.read_text()) if path.exists() else {}
    old.update(updates,updated_at=time.time())
    tmp=path.with_suffix('.tmp'); tmp.write_text(json.dumps(old,ensure_ascii=False,indent=2)); tmp.replace(path)

if __name__=='__main__':
    status(state='diagnosing',phase='配对检查：参考路径、收球策略和步态切换各自的影响',
        finding='参考路径单独自由物理模拟时，四种脚间距的俯身均保持平衡；完整侧移后仍失稳，正在分解切换因素。',
        metrics_text='上一轮完整流程0/32，摔倒14/32；本轮尚未宣称完整守门成功。',error=None)
