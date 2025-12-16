#!/usr/bin/env bash
set -euo pipefail

USER_ID="1858346102"
OUT_DIR="preview"

mkdir -p "$OUT_DIR"

echo "→ Récupération des IDs Deezer (liked tracks) pour USER_ID=$USER_ID ..."

# 1) Récupère tous les IDs (pagination via le champ 'next')
ids_file="$(mktemp)"
trap 'rm -f "$ids_file"' EXIT

url="https://api.deezer.com/user/${USER_ID}/tracks?limit=1000"

while [[ -n "$url" && "$url" != "null" ]]; do
  json="$(curl -fsS "$url")"
  echo "$json" | jq -r '.data[].id' >> "$ids_file"
  url="$(echo "$json" | jq -r '.next // empty')"
done

# Déduplique au cas où
sort -u "$ids_file" -o "$ids_file"

count="$(wc -l < "$ids_file" | tr -d ' ')"
echo "→ $count IDs trouvés. Téléchargement des previews (si disponibles)..."

# 2) Pour chaque ID, récupère la preview et télécharge
while IFS= read -r id; do
  out="${OUT_DIR}/${id}.mp3"

  # Si déjà téléchargé, on saute
  if [[ -s "$out" ]]; then
    echo "✓ $id déjà présent → skip"
    continue
  fi

  track_json="$(curl -fsS "https://api.deezer.com/track/${id}")"
  preview_url="$(echo "$track_json" | jq -r '.preview // empty')"

  if [[ -z "$preview_url" || "$preview_url" == "null" ]]; then
    echo "— $id : pas de preview disponible → skip"
    continue
  fi

  echo "↓ $id → téléchargement preview"
  curl -fLsS "$preview_url" -o "$out" || {
    echo "⚠️  $id : échec téléchargement (preview_url=$preview_url)" >&2
    rm -f "$out"
    continue
  }

  # petite pause pour rester poli avec l’API
  sleep 0.1
done < "$ids_file"

echo "✅ Terminé. Previews téléchargées dans ./${OUT_DIR}/"
