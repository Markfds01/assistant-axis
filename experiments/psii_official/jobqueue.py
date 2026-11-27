"""Job queue for the PSII reproduction: runs run_arm.sh jobs on 2 GPUs with several workers.
usage: queue.py JOBFILE WORKER_SPEC   (WORKER_SPEC "v0,0,1,1@900" etc;  e.g. ; a worker
named 'v0' only takes jobs whose arm is in VLLM_ARMS, on GPU0, then falls through to the general queue)
JOBFILE: lines 'arm seed'. Jobs are taken in file order. Skips jobs whose output already has 100 agent files."""
import glob, os, subprocess, sys, threading, time
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.environ.get("OUT", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "psii_outputs", "official"))
VLLM = {"direct", "pe"}
jobs = [l.split() for l in open(sys.argv[1]) if l.strip() and not l.startswith("#")]
lock = threading.Lock()
done_flag = threading.Event()

def finished(arm, seed):
    for d in glob.glob(f"{OUT}/{arm}/**/seed_{seed}", recursive=True):
        for dd in glob.glob(d + "/qstride_*"):
            if len(glob.glob(dd + "/*.json")) >= 100:
                return True
    return False

def take(only_vllm):
    with lock:
        for j in list(jobs):
            if only_vllm and j[0] not in VLLM:
                continue
            if (not only_vllm) and j[0] in VLLM and VLLM_WORKER_ALIVE[0]:
                continue
            jobs.remove(j)
            return j
    return None

VLLM_WORKER_ALIVE = [False]

def worker(name, gpu, vllm):
    if vllm:
        VLLM_WORKER_ALIVE[0] = True
    while True:
        j = take(vllm)
        if j is None:
            if vllm:
                VLLM_WORKER_ALIVE[0] = False
                vllm = False
                continue
            return
        arm, seed = j
        if finished(arm, seed):
            continue
        log = open(f"{OUT}/logs/{arm}_s{seed}.log", "w")
        env = dict(os.environ, QSTRIDE="4")
        print(time.strftime("%H:%M:%S"), name, "start", arm, seed, flush=True)
        subprocess.call([f"{HERE}/run_arm.sh", str(gpu), arm, seed], stdout=log, stderr=subprocess.STDOUT, env=env)
        print(time.strftime("%H:%M:%S"), name, "end", arm, seed, flush=True)

ths = []
VLLM_WORKER_ALIVE[0] = any(w.startswith("v") for w in sys.argv[2].split(","))
for i, w in enumerate(sys.argv[2].split(",")):
    w, _, delay = w.partition("@")
    vllm = w.startswith("v")
    gpu = int(w.lstrip("v"))
    t = threading.Thread(target=lambda i=i, gpu=gpu, vllm=vllm, delay=int(delay or 0): (time.sleep(delay), worker(f"w{i}", gpu, vllm))); t.start(); ths.append(t); time.sleep(90 if vllm else 20)
for t in ths:
    t.join()
