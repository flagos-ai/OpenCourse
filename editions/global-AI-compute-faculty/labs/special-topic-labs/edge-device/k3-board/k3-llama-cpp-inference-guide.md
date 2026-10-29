# K3 Pico-ITX llama.cpp Inference Lab Manual

This document is for the `llama.cpp` full model inference lab in the K3 Pico-ITX classroom. For Triton operator experiments, see `triton_operator_exercises.md`.

## 1. Prerequisites

The laptop needs to have:
- A Windows laptop or a macOS laptop.
- Ability to connect to the classroom Wi-Fi.

The K3 board needs to have:
- K3 Pico-ITX board.
- USB TTL serial cable.
- Type-C PD power supply.
- This round of class does not require installing an antenna.

---

## 2. Lab Flow Overview

This lab is strictly carried out in the following order:

1. Connect the development board.
2. Check whether the serial port is recognized, and according to the operating system (Windows / macOS), install the driver and configure the terminal.
3. Power on the K3.
4. Follow-up experiments: Wi-Fi connection, SSH login, llama.cpp inference and API experiments.

---

## 3. Step 1: Connect the Development Board (only need to connect the laptop side)

1. Connect the TX, RX, GND wires.
2. Plug the other side of the serial cable (USB end) into the laptop.

Serial cable colors:
```text
White
Green
Black
```
The red wire does not need to be connected.

---

## 4. Step 2: Check Serial Port Recognition and Configure Terminal (choose one according to operating system)

### 4.1 [Windows System Only]

**Step 2.1: Confirm whether the computer recognizes the COM port**

1. On the Windows desktop, right-click "This PC" -> "Manage" -> "Device Manager", or directly search for "Device Manager" in the search bar.
2. Expand `Ports (COM & LPT)`.

**Expected output / criteria:**
```text
See a device similar to the following (refer to screenshot):
USB-SERIAL CH340 (COM7)
```
> **Note:** The `COM7` in parentheses is the serial port number you need to select in MobaXterm. Your computer may be `COM3`, `COM7`, or another number, **please use the number actually displayed in your Device Manager**.

**Step 2.2: If not recognized, install the serial port driver**

1. If after expanding `Ports (COM & LPT)`, you do not see a device similar to `USB-SERIAL CH340`, or there is a yellow exclamation mark under the port, it means the serial cable connected to the laptop is not recognized and you need to download the driver.
2. Please open the following link in your laptop browser to download the driver:
   ```text
   https://www.wch-ic.com/downloads/CH341SER_EXE.html
   ```
3. After downloading, double-click to install, click "Next" all the way until the installation is complete.
4. After installation is complete, unplug and re-plug the USB end of the serial cable, then check Device Manager again.

**Expected output / criteria**: At this point you should normally see `USB-SERIAL CH340 (COMx)`.

**Step 2.3: Download and install MobaXterm**

1. Open a browser and visit the MobaXterm official download address:
   ```text
   https://mobaxterm.mobatek.net/download.html
   ```
2. Download the Home Edition Portable or Installer version.
3. After downloading, unzip or install, and open the software.

**Step 2.4: Configure serial session**

1. After opening MobaXterm, click `Sessions` in the menu bar.
2. Click `New Session`.

**Expected output / criteria:**
```text
The "Session settings" window pops up.
```

3. In the pop-up `Session settings` window, in the top row of icons, click `Serial`.
4. In the `Basic Serial settings` area:
   - `Serial port`: From the drop-down, select the COM port you just saw in Device Manager (such as `COM3` in the figure, please use your actual COM port).
   - `Speed (bps)`: Select `115200`.
5. In the `Advanced Serial settings` area, modify the following parameters:
   - `Flow control`: `None`
6. Click `OK` in the lower right corner.

**Expected output / criteria:**
```text
The Session window closes, and a new serial terminal tab appears in the main MobaXterm interface.
The terminal bottom may display "Connected" or a similar status, with no errors.
```

### 4.2 [macOS System Only]

**Step 2.1: Check serial devices**

1. Open Terminal and execute:
   ```bash
   ls /dev/tty.* /dev/cu.*
   ```

**Expected output / criteria:**
```text
You can see one of the following devices:
/dev/tty.usbserial-XXXX
/dev/cu.usbserial-XXXX
/dev/tty.SLAB_USBtoUART
/dev/cu.SLAB_USBtoUART
```
If there is no new device, check the USB serial driver or try another cable.

**Step 2.2: Use screen to open the serial port**

1. Use `screen` to open the serial port (replace `/dev/cu.usbserial-XXXX` with the device name actually seen in the previous step):
   ```bash
   screen /dev/cu.usbserial-XXXX 115200
   ```

**Expected output / criteria:**
```text
The terminal enters serial mode with no errors.
If it reports Cannot open line, check whether the device name is correct, and try /dev/tty.* if necessary.
```
> To exit `screen`: press `Ctrl+A`, release, then press `K`, then press `y` to confirm.

---

## 5. Step 3: Power on the K3 and Log in

1. Open the serial terminal (Windows MobaXterm window or macOS screen terminal), then **power on the K3**.

**Expected output / criteria:**
```text
The serial window starts showing boot logs.
Finally a login prompt appears, for example:
k3 login:
or
root@k3:~#
```

2. After seeing the login prompt, enter:
   ```text
   login: root
   Password: obtain the current classroom credential from the instructor.
   ```

**Expected output / criteria:**
```text
Successfully enter the shell, with a prompt similar to:
root@k3:~#
```

3. Check after login (common to all systems):
   ```bash
   whoami
   uname -m
   ```

**Expected output / criteria:**
```text
root
riscv64
```
If `uname -m` is not `riscv64`, it means you are not logged into the K3 target system, and you need to check the serial connection.

---

## 6. Follow-up Experiments (common to all systems)

Follow-up commands can be executed in either the serial terminal or the SSH terminal.

### 6.1 Connect to Wi-Fi

1. Execute in the K3 serial terminal:
   ```bash
   nmcli dev wifi list
   ```
   **Expected output / criteria:**
   ```text
   List nearby Wi-Fi, at least the classroom Wi-Fi SSID can be seen. No command errors.
   ```

2. Connect to the classroom Wi-Fi (this lab uniformly uses the following SSID and password):
   ```bash
   nmcli dev wifi connect "SSID" password "password"
   ```
   **Expected output / criteria:**
   ```text
   Something like: Device 'wlan0' successfully activated or Connection successfully activated
   ```
   If `Error` or `Secrets were required` appears, check the SSID and password.

3. Check K3 IP:
   ```bash
   ip addr
   ```
   **Expected output / criteria:**
   ```text
   Under wlan0 or a similar wireless network card, see: inet 192.168.x.x/24
   ```
   Record this IPv4 address for later SSH use.

### 6.2 SSH Login (common to all systems)

1. On the laptop's PowerShell (Windows) or Terminal (macOS), execute (replace `<BOARD_IP>`):
   ```bash
   ssh root@<BOARD_IP>
   ```
   **Expected output / criteria:**
   ```text
   First connection prompt: Are you sure you want to continue connecting (yes/no)? Enter yes.
   Then prompt: root@<BOARD_IP>'s password: enter the credential supplied for the board.
   After success, enter: root@k3:~#
   ```

### 6.3 Install llama.cpp Tools

1. Execute on K3:
   ```bash
   apt update
   ```
   **Expected output / criteria:**
   ```text
   Finally appears: Reading package lists... Done, and no errors starting with E:.
   ```

2. Install:
   ```bash
   apt install -y llama.cpp-tools-spacemit
   ```
   **Expected output / criteria:**
   ```text
   Finally appears: Setting up llama.cpp-tools-spacemit ..., no errors.
   ```

3. Verify installation:
   ```bash
   ls -l /usr/bin/llama-cli
   ls -l /usr/bin/llama-server
   test -x /usr/bin/llama-cli && echo "llama-cli: OK" || echo "llama-cli: MISSING"
   test -x /usr/bin/llama-server && echo "llama-server: OK" || echo "llama-server: MISSING"
   ```
   **Expected output / criteria:**
   ```text
   llama-cli: OK
   llama-server: OK
   ```
   If it shows `MISSING`, re-execute `apt update && apt install -y llama.cpp-tools-spacemit`.

### 6.4 Download Model

This round of class uses **Qwen3-0.6B-Q4_0 GGUF** by default.

1. Execute on K3:
   ```bash
   mkdir -p /root/models
   cd /root/models
   ```
   **Expected output / criteria:**
   ```text
   No error output. You can use pwd to confirm the current directory is /root/models.
   ```

2. Download:
   ```bash
   wget -O Qwen3-0.6B-Q4_0.gguf \
     "https://www.modelscope.cn/models/unsloth/Qwen3-0.6B-GGUF/resolve/master/Qwen3-0.6B-Q4_0.gguf"
   ```
   **Expected output / criteria:**
   ```text
   Download progress appears. Finally something like: ‘Qwen3-0.6B-Q4_0.gguf’saved
   ```
   If it prompts `wget: command not found`, first execute `apt install -y wget`.

3. Check file:
   ```bash
   ls -lh /root/models/Qwen3-0.6B-Q4_0.gguf
   ```
   **Expected output / criteria:**
   ```text
   Displays the file, size is not 0. Qwen3-0.6B-Q4_0 is about a few hundred MB.
   ```
   Linux file names are case-sensitive, and the case in the command must remain consistent.

### 6.5 llama-cli Inference Experiment

According to your request, only modify the `6.5 llama-cli Inference Experiment` section:

- Change `-n 32` to `-n -1` to avoid output being truncated;
- Add parameter descriptions;
- Do not change anything else.

The following is the modified `6.5` subsection, other sections remain unchanged:

---

### 6.5 llama-cli Inference Experiment

1. Execute on K3:
   ```bash
   /usr/bin/llama-cli \
     -m /root/models/Qwen3-0.6B-Q4_0.gguf \
     -p "Explain edge AI in one sentence." \
     -n -1 \
     -t 4
   ```

   **Parameter descriptions:**
   - `-m /root/models/Qwen3-0.6B-Q4_0.gguf`: Specify the model file path.
   - `-p "Explain edge AI in one sentence."`: Prompt, the model generates content based on it.
   - `-n -1`: Maximum number of tokens to generate. `-1` means no limit, until the model naturally ends or you press `Ctrl+C` to stop.
   - `-t 4`: Number of CPU threads used.

   **Expected output / criteria:**
   ```text
   1. First, CPU_RISCV64_SPACEMIT related information appears. May include:
      alloc_chunk: open(/dev/tcm_sync_mem) failed
      failed to allocate init_barrier from shared mem, falling back to heap
      This is a normal fallback warning and usually does not affect inference.
   2. The llama.cpp banner and model information appear (model: /root/models/Qwen3-0.6B-Q4_0.gguf).
   3. The model outputs a short English passage; the generated content is not required to be verbatim.
   4. Finally something like: [ Prompt: 272.3 t/s | Generation: 52.4 t/s ]
   ```
   Judged as correct: no `failed to load model` or `error loading model`, actual generated text, and t/s statistics.

**Complete example output (from actual test, for format and interaction reference only, generated content is not required to be verbatim):**
```text
(.triton-venv) root@k3:~/llama.cpp# /usr/bin/llama-cli \
  -m /root/models/Qwen3-0.6B-Q4_0.gguf \
  -n -1 \
  -t 4
CPU_RISCV64_SPACEMIT: tcm is available, blk_size: 393216, blk_num: 8, is_fake_tcm: 0
CPU_RISCV64_SPACEMIT: num_cores: 16, num_perfer_cores: 8, perfer_core_arch_id: a064, exclude_main_thread: 0, use_ime1: 0, use_ime2: 1, mem_backend: HPAGE, cpu_mask: ff00, aicpu_id_offset: 8
CPU_RISCV64_SPACEMIT: alloc_chunk: open(/dev/tcm_sync_mem) failed, errno=2
CPU_RISCV64_SPACEMIT: failed to allocate init_barrier from shared mem, falling back to heap


Loading model...

▄▄ ▄▄
██ ██
██ ██  ▀▀█▄ ███▄███▄  ▀▀█▄    ▄████ ████▄ ████▄
██ ██ ▄█▀██ ██ ██ ██ ▄█▀██    ██    ██ ██ ██ ██
██ ██ ▀█▄██ ██ ██ ██ ▀█▄██ ██ ▀████ ████▀ ████▀
                                    ██    ██
                                    ▀▀    ▀▀

build      : b10050-5496cdc9c
model      : /root/models/Qwen3-0.6B-Q4_0.gguf
ftype      : Q4_0
modalities : text

available commands:
  /exit or Ctrl+C     stop or exit
  /regen              regenerate the last response
  /clear              clear the chat history
  /read <file>        add a text file
  /glob <pattern>     add text files using globbing pattern



> hello
Hello! How can I assist you today?

[ Prompt: 272.3 t/s | Generation: 52.4 t/s ]

> what is triton?
Triton is a distributed system for machine learning that is designed to scale horizontally. It allows you to create scalable and efficient models that can handle large amounts of data and perform complex tasks. Triton was developed by Alibaba Cloud and is built on top of the popular TensorFlow framework. It is designed to be easy to use and provides a simple API that allows developers to build scalable and efficient models with minimal code.

One of the key features of Triton is that it can handle very large datasets, which is a critical requirement for many machine learning tasks. Triton is designed to be able to handle millions of samples without requiring expensive computational resources.

Triton's distributed architecture allows for efficient data processing and can be used in conjunction with other distributed machine learning frameworks such as Apache Hadoop and Apache Spark, which further expands its capabilities.

In summary, Triton is a distributed system for machine learning that provides a scalable and efficient way to build and train machine learning models with very large datasets.

[ Prompt: 199.9 t/s | Generation: 44.2 t/s ]

> what is LLM?
LLM stands for Large Language Model. A LLM is a machine learning model that is designed to generate human-like language output. These models are trained using large amounts of text data, which allows them to understand and generate natural language in a way that is very human-like.

There are two main types of LLMs: those that are "translating" models and those that are "generating" models. A translating model is a model that is trained to generate human-like language output, while a generating model is a model that is trained to generate human-like language output.

One of the key features of LLMs is that they can generate human-like language output. This is a critical requirement for many machine learning tasks, and LLMs are used to train other machine learning models that can perform tasks such as text summarization, language translation, and natural language understanding.

LLMs are used in a wide range of applications, including natural language processing, AI, and robotics. They are used in various domains, such as language modeling, language translation, text summarization, language generation, and natural language inference.

In summary, an LLM is a machine learning model that is designed to generate human-like language output. LLMs are used to train other machine learning models that can perform tasks such as text summarization, language translation, text summarization, language generation, and natural language inference.

[ Prompt: 74.5 t/s | Generation: 33.0 t/s ]

> what can you do?
As a powerful assistant, I can perform a wide range of tasks, including:

1. Answering questions to the best of my ability and to the best of my knowledge.
2. Providing information on a wide range of topics, including technology, science, history, and more.
3. Checking the accuracy of documents and data, and providing corrections when necessary.
4. Assisting with research and data gathering, as well as generating summaries and reports.
5. Handling and analyzing large datasets, including data cleaning and preprocessing, data visualization, and data analysis.
6. Providing guidance and suggestions on how to improve or enhance a specific aspect of a project or application.
7. Helping with project planning and execution, such as creating project timelines and scheduling, and providing guidance on resource allocation.
8. Providing support for specific use cases or use scenarios, such as answering questions about a particular use case or use scenario, or suggesting alternative solutions to a problem.
9. Providing advice and guidance on how to approach new challenges or situations, and helping to identify potential opportunities for improvement.

I can also assist with other tasks, including but not limited to:

10. Assisting with writing and editing tasks, such as proofreading documents, providing feedback on writing, and assisting with other forms of writing.
11. Providing advice and guidance on how to improve or enhance a specific aspect of a project or application, such as suggesting ways to improve performance, optimize resource usage, or enhance user experience.
12. Assisting with creating and maintaining documentation and guides, such as providing guidance on how to use a new feature or how to troubleshoot a specific issue.
13. Providing guidance and suggestions on how to improve or enhance a specific aspect of a project or application, such as suggesting ways to improve performance, optimize resource usage, or enhance user experience.
14. Assisting in the development and management of codebases, providing guidance on how to write and maintain code, and providing advice on best practices and guidelines for writing code.
15. Assisting with other forms of creative or artistic work, such as providing guidance and suggestions on how to improve or enhance a specific aspect of a particular form of art or creative expression, or assisting with the development and management of a specific type of art or creative expression.

I can also help with tasks that require collaboration, such as providing guidance and suggestions on how to improve or enhance a specific aspect of a project or application, and assisting with the development and management of a specific type of project or application.
17. Assisting with other forms of support and assistance, such as providing guidance and suggestions on how to improve or enhance a specific aspect of a project or application, and assisting with the development and management of a specific type of project or application.

In summary, LLMs have a wide range of applications and can be used in a variety of tasks, including answering questions, generating text, providing guidance and suggestions, and assisting with project management and project management tasks. I am designed to be able to assist with a wide range of tasks, including those related to LLMs and other machine learning techniques.

[ Prompt: 53.8 t/s | Generation: 26.9 t/s ]

>

Exiting...
(.triton-venv) root@k3:~/llama.cpp# ^C
```

If it enters interactive mode and you see `>`, exit method: enter `/exit` or press `Ctrl+C`.
**Expected output / criteria:**
```text
The process exits and returns to the shell prompt, for example: root@k3:~#
```

### 6.6 llama-server API Experiment

1. Start the server in one K3 terminal:
   ```bash
   /usr/bin/llama-server \
     -m /root/models/Qwen3-0.6B-Q4_0.gguf \
     --host 127.0.0.1 \
     --port 8080
   ```
   **Expected output / criteria:**
   ```text
   Something like: llama server listening at http://127.0.0.1:8080
   And the process keeps running, does not exit automatically.
   ```
   Keep this terminal open.

2. Check health status in another K3 terminal:
   ```bash
   curl http://127.0.0.1:8080/health
   ```
   **Expected output / criteria:**
   ```text
   Returns JSON, something like: {"status":"ok"}
   If it returns {"status":"loading model"}, it means the model is still loading, wait and retry.
   ```
   If it prompts `curl: command not found`, first execute `apt install -y curl`.

3. Call Chat Completions API:
   ```bash
   curl -X POST http://127.0.0.1:8080/v1/chat/completions \
     -H "Content-Type: application/json" \
     -d '{
       "messages": [
         {"role": "user", "content": "Explain edge AI in one sentence."}
       ],
       "max_tokens": 32
     }'
   ```
   **Expected output / criteria:**
   ```text
   Returns JSON. It contains a choices array. choices[0].message.content has the model-generated English text. No error field.
   ```
   Example structure:
   ```json
   {
     "choices": [
       {
         "message": {
           "role": "assistant",
           "content": "Edge AI means running AI models locally on devices instead of in the cloud."
         }
       }
     ]
   }
   ```

4. In a browser, visit `http://127.0.0.1:8080`.
   **Expected output / criteria:**
   ```text
   Open the llama-server web interface, the page displays normally, you can enter messages and get replies.
   ```

---

## 7. Common Problems

| Symptom | Solution |
|---|---|
| After plugging in the serial cable on Windows, Device Manager does not recognize it | Go to `https://www.wch-ic.com/downloads/CH341SER_EXE.html` to download and install the CH340 driver, then unplug and re-plug |
| Device Manager has a COM port, but MobaXterm cannot select it | Close MobaXterm and reopen it, or manually enter the COM port in MobaXterm's Serial settings |
| macOS cannot find the serial port | Execute `ls /dev/tty.* /dev/cu.*` to confirm the driver; use the correct `/dev/cu.*` device name |
| macOS `screen` cannot open the serial port | Check the device name, try `/dev/cu.*`; if necessary use `sudo screen ...` |
| MobaXterm has no output | Confirm serial cable is connected first then power on, confirm `115200 8N1 None flow control` |
| Login failed | Confirm the classroom credential, then check keyboard layout and Caps Lock |
| Wi-Fi connection failed | Rescan SSID, confirm password, this time SSID is `ZhongZhiYuan`, password is `12345678` |
| `llama-cli: MISSING` | Execute `apt update && apt install -y llama.cpp-tools-spacemit` |
| `llama-server: MISSING` | Execute `apt update && apt install -y llama.cpp-tools-spacemit` |
| `wget: command not found` | Execute `apt install -y wget` |
| `curl: command not found` | Execute `apt install -y curl` |
| Model does not exist | Re-execute the ModelScope `wget` download command, confirm the file is `/root/models/Qwen3-0.6B-Q4_0.gguf` |
| Model loading failed | Check model path and file name case, must be `/root/models/Qwen3-0.6B-Q4_0.gguf` |
| `open(/dev/tcm_sync_mem) failed` / `falling back to heap` | Normal fallback, usually does not affect inference; continue to observe whether the model can load and generate text |
| `/health` failed | Confirm `llama-server` is still running, wait for model loading to complete and retry |
| `/health` returns loading | Model is still loading, wait and retry |

## 8. Shutdown Sequence

1. If `llama-server` is running, first press `Ctrl+C` to stop it.
   **Expected output / criteria:**
   ```text
   The server process exits and returns to the shell.
   ```

2. Then execute:
   ```bash
   sync
   poweroff
   ```
   **Expected output / criteria:**
   ```text
   sync usually has no output. After poweroff, the serial or SSH disconnects, and the board stops running.
   ```
   Wait until the serial or SSH disconnects and the board is stable, then disconnect the power.
