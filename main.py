import os, json, time, subprocess
import numpy as np
import requests
import torch
from tqdm import tqdm
import soundfile as sf
import librosa
from panns_inference import AudioTagging

from transformers import AutoProcessor, AutoModel, AutoModelForAudioClassification

# -----------------------
# Config
# -----------------------
USER_ID = "1858346102"
OUT_DIR = "preview"
EMB_PATH = "embeddings.npy"
META_PATH = "embeddings_meta.jsonl"
LIMIT = 1000
SLEEP_SECONDS = 0.15
TARGET_SR = 16000
TOP_K = 15

os.makedirs(OUT_DIR, exist_ok=True)

# -----------------------
# Models
# -----------------------
MODEL_ID = "MIT/ast-finetuned-audioset-10-10-0.4593"

device = "cuda" if torch.cuda.is_available() else "cpu"

# Model 1 (AST)
processor = AutoProcessor.from_pretrained(MODEL_ID)
embed_model = AutoModel.from_pretrained(MODEL_ID).eval().to(device)
clf_model = AutoModelForAudioClassification.from_pretrained(MODEL_ID).eval().to(device)
id2label = clf_model.config.id2label  # {id: label}

# Model 2 (PANNs)
# PANNs gère son propre device, on lui passe 'cuda' ou 'cpu'
panns_model = AudioTagging(checkpoint_path=None, device=device)

# -----------------------
# Deezer helpers
# -----------------------
session = requests.Session()
session.headers.update({"User-Agent": "deezer-ast-embedder/1.0"})

def get_json(url, timeout=10, max_retries=4):
    for attempt in range(max_retries):
        try:
            r = session.get(url, timeout=timeout)
            r.raise_for_status()
            return r.json()
        except Exception:
            if attempt == max_retries - 1:
                raise
            time.sleep(0.5 * (2 ** attempt))

def get_all_track_ids(user_id):
    ids = []
    url = f"https://api.deezer.com/user/{user_id}/tracks?limit={LIMIT}"
    while url:
        data = get_json(url)
        ids.extend(t["id"] for t in data.get("data", []))
        url = data.get("next")
    return sorted(set(ids))

def get_track_info(track_id):
    d = get_json(f"https://api.deezer.com/track/{track_id}")
    return {
        "id": track_id,
        "title": d.get("title"),
        "artist": (d.get("artist") or {}).get("name"),
        "album": (d.get("album") or {}).get("title"),
        "preview": d.get("preview"),
        "link": d.get("link"),
    }

def download_file(url, out_path):
    if os.path.exists(out_path) and os.path.getsize(out_path) > 0:
        return
    with session.get(url, stream=True, timeout=15) as r:
        r.raise_for_status()
        with open(out_path, "wb") as f:
            for chunk in r.iter_content(8192):
                if chunk:
                    f.write(chunk)

def mp3_to_wav(mp3_path, wav_path):
    cmd = ["ffmpeg", "-y", "-i", mp3_path, "-ac", "1", "-ar", str(TARGET_SR), wav_path]
    p = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    if p.returncode != 0:
        raise RuntimeError("ffmpeg conversion failed (install ffmpeg & add to PATH)")

# -----------------------
# Audio loading
# -----------------------
def load_wav_mono(path, target_sr=16000):
    wav, sr = sf.read(path)
    if wav.ndim == 2:
        wav = wav.mean(axis=1)
    if sr != target_sr:
        wav = librosa.resample(wav, orig_sr=sr, target_sr=target_sr)
        sr = target_sr
    return wav.astype(np.float32), sr

# -----------------------
# Embedding + preds
# -----------------------
@torch.no_grad()
def ast_embed_and_preds(waveform_np, sr, top_k=15):
    # --- Model 1 (AST) ---
    # AST attend du 16kHz (géré par load_wav_mono)
    inputs = processor(waveform_np, sampling_rate=sr, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    out_emb = embed_model(**inputs)
    emb1 = out_emb.last_hidden_state.mean(dim=1).squeeze(0)  # (768,)
    emb1 = torch.nn.functional.normalize(emb1, dim=0)

    # --- Model 2 (PANNs) ---
    # PANNs attend du 32kHz
    if sr != 32000:
        wav_32k = librosa.resample(waveform_np, orig_sr=sr, target_sr=32000)
    else:
        wav_32k = waveform_np
    
    # Ajout dimension batch : (1, samples)
    wav_32k = wav_32k[None, :]
    
    _, emb2_np = panns_model.inference(wav_32k) # emb2_np est (1, 2048)
    emb2 = torch.from_numpy(emb2_np).squeeze(0).to(device)
    emb2 = torch.nn.functional.normalize(emb2, dim=0)

    # --- Fusion ---
    emb_cat = torch.cat([emb1, emb2], dim=0)
    emb_np = emb_cat.detach().cpu().numpy()

    # Multi-label probs (AST)
    out_clf = clf_model(**inputs)
    probs = torch.sigmoid(out_clf.logits.squeeze(0))  # (num_labels,)
    top = torch.topk(probs, k=top_k)

    preds = [
        {"label": id2label[int(i)], "score": float(s)}
        for s, i in zip(top.values.detach().cpu().tolist(), top.indices.detach().cpu().tolist())
    ]

    return emb_np, preds

# -----------------------
# Persistence
# -----------------------
def load_done_ids(meta_path):
    done = set()
    if not os.path.exists(meta_path):
        return done
    with open(meta_path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                done.add(json.loads(line)["id"])
            except Exception:
                pass
    return done

def append_meta(meta_path, obj):
    with open(meta_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")

# -----------------------
# Main
# -----------------------
def main():
    track_ids = get_all_track_ids(USER_ID)
    done = load_done_ids(META_PATH)
    todo = [tid for tid in track_ids if tid not in done]
    print(f"{len(track_ids)} titres | {len(todo)} à traiter")

    existing = None
    if os.path.exists(EMB_PATH):
        existing = np.load(EMB_PATH)
        print("Embeddings existants:", existing.shape)

    new_embs = []

    for tid in tqdm(todo):
        mp3_path = os.path.join(OUT_DIR, f"{tid}.mp3")
        wav_path = os.path.join(OUT_DIR, f"{tid}.wav")

        try:
            info = get_track_info(tid)
            if not info["preview"]:
                append_meta(META_PATH, {**info, "error": "no_preview"})
                continue

            download_file(info["preview"], mp3_path)
            mp3_to_wav(mp3_path, wav_path)

            wav_np, sr = load_wav_mono(wav_path, TARGET_SR)
            emb, preds = ast_embed_and_preds(wav_np, sr, top_k=TOP_K)

            # 1) on écrit la meta (humain)
            append_meta(META_PATH, {**info, "dim": int(emb.shape[0]), "preds": preds})

            # 2) on écrit l'embedding (machine)
            new_embs.append(emb)

        except Exception as e:
            append_meta(META_PATH, {"id": tid, "error": str(e)})

        finally:
            for p in (mp3_path, wav_path):
                if os.path.exists(p):
                    os.remove(p)

        time.sleep(SLEEP_SECONDS)

    if not new_embs:
        print("Aucun nouvel embedding")
        return

    new_mat = np.vstack(new_embs)

    if existing is None:
        np.save(EMB_PATH, new_mat)
        print("Sauvé:", new_mat.shape)
    else:
        merged = np.vstack([existing, new_mat])
        np.save(EMB_PATH, merged)
        print("Sauvé:", merged.shape)

if __name__ == "__main__":
    main()
