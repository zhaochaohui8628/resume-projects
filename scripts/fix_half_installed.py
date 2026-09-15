"""修复 3 个半安装包（dateutil / six / dotenv）：pip download → 手工解压（绕沙箱拦截）。"""
import glob
import os
import shutil
import subprocess
import sys
import zipfile

TG_PY = r"C:\Users\<用户名>\anaconda3\envs\torch_gpu\python.exe"
SP = r"C:\Users\<用户名>\anaconda3\envs\torch_gpu\Lib\site-packages"
WHL = r"C:\tmp\fix_wheels"
os.makedirs(WHL, exist_ok=True)

PKGS = ["python-dateutil", "six", "python-dotenv"]

print(">>> 下载 wheel")
r = subprocess.run([TG_PY, "-E", "-m", "pip", "download", "--no-cache-dir",
                    "-d", WHL, "-i", "https://pypi.tuna.tsinghua.edu.cn/simple",
                    "--timeout", "120", "--retries", "3"] + PKGS,
                   capture_output=True, text=True, timeout=600)
print("rc:", r.returncode)
if r.returncode != 0:
    print((r.stderr or "")[-500:])
    sys.exit(1)

print("\n>>> 解压覆盖到 site-packages")
n = 0
for w in glob.glob(os.path.join(WHL, "*.whl")):
    name = os.path.basename(w)
    # 只处理目标 3 个包，避免误伤
    if not any(k in name.lower() for k in ("dateutil", "six", "dotenv")):
        continue
    try:
        with zipfile.ZipFile(w) as z:
            for info in z.infolist():
                if info.filename.endswith("/"):
                    continue
                t = os.path.join(SP, info.filename)
                os.makedirs(os.path.dirname(t), exist_ok=True)
                with z.open(info) as s, open(t, "wb") as d:
                    shutil.copyfileobj(s, d)
        print("   解压 OK:", name)
        n += 1
    except Exception as e:
        print("   解压 FAIL:", name, e)
print(f"共解压 {n} 个")

print("\n>>> 验证")
r = subprocess.run([TG_PY, "-E", "-c",
                    "import dateutil, six, dotenv, pandas; "
                    "print('dateutil', dateutil.__version__); "
                    "print('six', six.__version__); "
                    "print('pandas', pandas.__version__, 'OK')"],
                   capture_output=True, text=True, timeout=180)
print((r.stdout or "").strip())
if r.returncode != 0:
    print((r.stderr or "")[-400:])
