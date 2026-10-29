# FlagOS / BI-V150 workflow

Clone the current course repository from GitLink and enter the Module 5 lab:

```bash
git clone https://gitlink.org.cn/flagos-ai/OpenCourse.git
cd OpenCourse/labs/modules/module-05/iluvatar-corex-bi-v150
```

Run the workflow on the BI-V150 environment:

```bash
source env_corex.sh
bash setup.sh
python3 00_check_env.py
python3 00_smoke_cuda.py
python3 01_benchmark.py --scenario offline --repeats 3
python3 01_benchmark.py --scenario server --repeats 3
python3 02_roofline.py
python3 03_profiling.py --capture
python3 04_summary.py
```

If Roofline plotting is needed, install `matplotlib`; the JSON report is still
produced when the package is unavailable. The lab README is the authoritative
guide to result interpretation and acceptance requirements.
