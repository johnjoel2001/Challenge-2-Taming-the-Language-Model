# RunPod Setup Guide

How to connect to a RunPod GPU instance from VSCode terminal, move files, and avoid losing data.

---

## 1. Create a Pod and Set Up SSH

1. Go to [runpod.io](https://www.runpod.io) → **Pods** → **Deploy**
2. Choose a GPU template (e.g. PyTorch with CUDA)
3. **Attach a Network Volume** before deploying (see section 3 — do this first)
4. Under **Advanced** settings, paste your public SSH key:
   ```bash
   # Get your public key on your local machine
   cat ~/.ssh/id_rsa.pub
   ```
   Paste the output into the RunPod SSH key field.

5. Deploy the pod. Once it's running, click **Connect** → **SSH over exposed TCP** to get the exact SSH command — it looks like:
   ```
   ssh root@<ip-address> -p <port> -i ~/.ssh/id_rsa
   ```

---

## 2. SSH Into the Pod (VSCode Terminal)

Open the VSCode integrated terminal (`Ctrl+`` ` or `Cmd+`` `) and run:

```bash
ssh root@<ip-address> -p <port> -i ~/.ssh/id_rsa
```

Replace `<ip-address>` and `<port>` with values from the RunPod Connect tab.

To avoid typing this every time, add an entry to `~/.ssh/config`:

```
Host runpod
    HostName <ip-address>
    Port <port>
    User root
    IdentityFile ~/.ssh/id_rsa
```

Then just run:
```bash
ssh runpod
```

---

## 3. Network Volume — Store Data Here, Not on Disk

**Pod disk is ephemeral.** Anything saved outside the network volume is deleted when the pod stops or is terminated.

**Network volume persists** across pod restarts and terminations.

### Create a Network Volume
1. RunPod dashboard → **Storage** → **New Network Volume**
2. Choose the same region as your pod
3. Give it a name and size (e.g. 20 GB)
4. When deploying a pod, attach this volume — it mounts at `/workspace`

### Always save to `/workspace`
```bash
# On the pod — store models, data, outputs here
cd /workspace
mkdir -p Challenge-2-Taming-the-Language-Model
```

Set the output path in your scripts to `/workspace/...` or clone/copy the project there:
```bash
git clone <your-repo-url> /workspace/Challenge-2-Taming-the-Language-Model
cd /workspace/Challenge-2-Taming-the-Language-Model
```

---

## 4. Transferring Files with `scp`

`scp` (secure copy) uses SSH to transfer files. The `-P` flag (capital P) sets the port.

### Copy a file from local → pod
```bash
scp -P <port> -i ~/.ssh/id_rsa ./local_file.py root@<ip-address>:/workspace/
```

### Copy a directory from local → pod
```bash
scp -P <port> -i ~/.ssh/id_rsa -r ./src/ root@<ip-address>:/workspace/Challenge-2-Taming-the-Language-Model/
```

### Copy a file from pod → local
```bash
scp -P <port> -i ~/.ssh/id_rsa root@<ip-address>:/workspace/outputs/eval_metrics_summary.csv ./outputs/
```

### Copy a directory from pod → local
```bash
scp -P <port> -i ~/.ssh/id_rsa -r root@<ip-address>:/workspace/outputs/figures/ ./outputs/
```

If you set up the `~/.ssh/config` alias above, shorten to:
```bash
scp -r runpod:/workspace/outputs/figures/ ./outputs/
```

---

## 5. Syncing Files with `rsync`

`rsync` is faster than `scp` for repeated transfers — it only sends changed files.

```bash
# Push local src/ to pod
rsync -avz -e "ssh -p <port> -i ~/.ssh/id_rsa" ./src/ root@<ip-address>:/workspace/Challenge-2-Taming-the-Language-Model/src/

# Pull outputs back
rsync -avz -e "ssh -p <port> -i ~/.ssh/id_rsa" root@<ip-address>:/workspace/outputs/ ./outputs/
```

---

## 6. Running the Pipeline on the Pod

Once SSH'd in and files are in `/workspace`:

```bash
cd /workspace/Challenge-2-Taming-the-Language-Model
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt

export RLHF_MODE=extended
python3 -m src.generate_baseline && \
python3 -m src.create_preference_data && \
python3 -m src.train_reward_model_simplicity && \
python3 -m src.train_reward_model_balanced && \
python3 -m src.train_ppo_simplicity && \
python3 -m src.train_ppo_balanced && \
python3 -m src.evaluate && \
python3 -m src.misalignment_analysis && \
python3 -m src.plotting
```

To run in the background so it survives SSH disconnects:
```bash
nohup bash run_all.sh > /workspace/run.log 2>&1 &
tail -f /workspace/run.log   # watch progress
```

---

## 7. Quick Reference

| Task | Command |
|---|---|
| SSH into pod | `ssh root@<ip> -p <port> -i ~/.ssh/id_rsa` |
| Copy file to pod | `scp -P <port> -i ~/.ssh/id_rsa file root@<ip>:/workspace/` |
| Copy file from pod | `scp -P <port> -i ~/.ssh/id_rsa root@<ip>:/workspace/file ./` |
| Copy directory to pod | `scp -P <port> -i ~/.ssh/id_rsa -r dir/ root@<ip>:/workspace/` |
| Sync to pod | `rsync -avz -e "ssh -p <port>" dir/ root@<ip>:/workspace/dir/` |
| Watch logs | `tail -f /workspace/run.log` |
| Check GPU | `nvidia-smi` |
| Run in background | `nohup python3 script.py > out.log 2>&1 &` |

**Remember:** Always store files in `/workspace/` — anything on the pod disk is lost when the pod stops.
