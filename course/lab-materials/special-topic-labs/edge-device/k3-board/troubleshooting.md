# Troubleshooting

## Serial console

### No serial port appears on Windows

First actions:

1. Replug the USB serial cable.
2. Check Windows Device Manager.
3. Install the USB serial driver provided by the instructor.
4. Try another USB port or another cable.

### Serial window opens but shows nothing

First actions:

1. Confirm the speed is `115200`.
2. Confirm MobaXterm uses `8` data bits, `1` stop bit, `None` parity, and `None` flow control.
3. Press Enter once or twice.
4. Confirm the board is powered.
5. Ask the instructor to check the UART cable orientation.

The classroom connection order is: connect the serial cable first, then power on the K3 board. The classroom cable colors follow the instructor's verified marking: white, green, black; red is not connected.

The current reference material confirms `TX/RX/GND` and `115200`, but it does not clearly document UART voltage or the physical pin order of the debug connector. Do not guess these in class; ask the instructor to verify against the prepared hardware photo.

## Login

Classroom account:

- Username: `root`
- Password: obtain the current classroom credential from the instructor.

If login fails, check keyboard layout, Caps Lock, and whether you are connected to the correct board.

## Wi-Fi and SSH

### Wi-Fi network is not visible

```bash
nmcli dev wifi list
```

First actions:

1. Refresh the list after a few seconds.
2. Confirm the SSID with the instructor.
3. Ask the instructor whether to use the backup hotspot/router.

### Wi-Fi connects but SSH fails

First actions:

1. Check the board IP address:

   ```bash
   ip addr
   ```

2. Make sure the laptop is on the same Wi-Fi network.
3. Classroom Wi-Fi may block client-to-client traffic. If so, continue through serial or use the instructor's hotspot/router.

## Model download and checks

### Model download fails

The classroom target path is:

```text
/root/models/Qwen3-0.6B-Q4_0.gguf
```

The classroom download command is:

```bash
mkdir -p /root/models
cd /root/models
wget -O Qwen3-0.6B-Q4_0.gguf \
  "https://www.modelscope.cn/models/unsloth/Qwen3-0.6B-GGUF/resolve/master/Qwen3-0.6B-Q4_0.gguf"
```

First actions:

1. Confirm Wi-Fi is connected.
2. Confirm DNS and general network access work.
3. Copy the full URL again; do not omit quotation marks.
4. Check available disk space:

   ```bash
   df -h
   ```

5. If the file is incomplete, delete it and download again.

After download, check the file size and checksum if the instructor provided a reference:

```bash
ls -lh /root/models/Qwen3-0.6B-Q4_0.gguf
sha256sum /root/models/Qwen3-0.6B-Q4_0.gguf
```

The download is not considered verified until the file passes size/checksum checks and loads successfully with `llama-cli`.

### `llama-cli: MISSING`

Check whether the command exists:

```bash
ls -l /root/llama.cpp/build/bin/llama-cli
```

If the instructor prepared the SpacemiT system package version, the classroom path may be a symbolic link to `/usr/bin/llama-cli`. Ask the instructor to restore the prepared link or command path.

### `llama-server: MISSING`

Check whether the command exists:

```bash
ls -l /root/llama.cpp/build/bin/llama-server
```

If it is missing, ask the instructor whether the board image includes `llama-server`. Do not spend classroom time rebuilding `llama.cpp` unless the instructor tells you to.

### Model file not found

Use the classroom model path exactly:

```text
/root/models/Qwen3-0.6B-Q4_0.gguf
```

Linux filenames are case-sensitive. Copy the path from the lab handout instead of retyping it from memory.

### Model is corrupted or incomplete

A typical error may mention that tensor data is outside file bounds, for example:

```text
tensor 'token_embd.weight' data is not within the file bounds, model is corrupted or incomplete
```

First actions:

1. Stop the command.
2. Confirm the model path is `/root/models/Qwen3-0.6B-Q4_0.gguf`.
3. Delete the incomplete file.
4. Re-run the ModelScope `wget` command.
5. Recheck file size and SHA256 if the instructor provided a reference.

## Inference commands

### `llama-cli` enters interactive mode

If you see a prompt such as `>`, you can type a question and press Enter.

To exit:

```text
/exit
```

or press `Ctrl+C`.

### Generation is too slow

Use a shorter prompt and a smaller token limit, for example `-n 32`. If the class is behind schedule, the instructor may switch this part to a demo.

### `curl http://127.0.0.1:8080/health` fails

First actions:

1. Confirm `llama-server` is still running in another terminal.
2. Confirm the server was started with `--port 8080`.
3. Wait a few seconds after server startup, then retry.
4. If the server printed an error while loading the model, fix the model path first.

### `/health` returns `Loading model` or HTTP 503

This can be normal while `llama-server` is loading the GGUF model. Confirm that the server is still running, wait for the model-loading message to finish, and retry:

```bash
curl http://127.0.0.1:8080/health
```

The current upstream API commonly returns an OK-like response after the model is ready. The exact response can vary by `llama.cpp` version.

## Important note about acceleration

In this workshop, `llama.cpp` is the complete-model inference experience on the K3 board. Do not claim it uses the K3 AI accelerator unless that path is separately verified and documented.

## Triton scripts

Triton and spine-triton experiments are intentionally not part of the current inference-only lab flow. If you see older files such as `verify_install.py`, `vec_add.py`, or `softmax_demo.py`, keep them for future use but do not use them as the main classroom checklist for this version.
