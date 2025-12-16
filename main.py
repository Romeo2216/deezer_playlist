import os
import time
import requests

USER_ID = "1858346102"
OUT_DIR = "preview"
LIMIT = 1000
SLEEP_SECONDS = 0.1  # pour rester poli avec l’API Deezer

os.makedirs(OUT_DIR, exist_ok=True)


def get_all_track_ids(user_id):
    """Récupère tous les IDs des musiques aimées (pagination Deezer)."""
    ids = []
    url = f"https://api.deezer.com/user/{user_id}/tracks?limit={LIMIT}"

    while url:
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        data = resp.json()

        ids.extend(track["id"] for track in data.get("data", []))
        url = data.get("next")

    return sorted(set(ids))


def get_preview_url(track_id):
    """Récupère l’URL de preview (30s) pour un track donné."""
    url = f"https://api.deezer.com/track/{track_id}"
    resp = requests.get(url, timeout=10)
    resp.raise_for_status()
    return resp.json().get("preview")


def download_preview(track_id, preview_url):
    """Télécharge la preview MP3 dans le dossier preview/."""
    output_path = os.path.join(OUT_DIR, f"{track_id}.mp3")

    if os.path.exists(output_path):
        print(f"✓ {track_id} déjà téléchargé")
        return

    with requests.get(preview_url, stream=True, timeout=10) as r:
        r.raise_for_status()
        with open(output_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)

    print(f"↓ {track_id} preview téléchargée")


def main():
    print("→ Récupération des IDs des musiques aimées…")
    track_ids = get_all_track_ids(USER_ID)
    print(f"→ {len(track_ids)} titres trouvés")

    for track_id in track_ids:
        try:
            preview_url = get_preview_url(track_id)

            if not preview_url:
                print(f"— {track_id} : pas de preview disponible")
                continue

            download_preview(track_id, preview_url)
            time.sleep(SLEEP_SECONDS)

        except Exception as e:
            print(f"⚠️  {track_id} : erreur ({e})")


if __name__ == "__main__":
    main()
