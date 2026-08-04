#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-labels}"
DESTINATION="${2:-data/raw/bright}"
RECORD="https://zenodo.org/records/20072020/files"

mkdir -p "${DESTINATION}"
download() {
  local name="$1"
  if [[ ! -f "${DESTINATION}/${name}" ]]; then
    curl -L --fail --show-error --output "${DESTINATION}/${name}" "${RECORD}/${name}?download=1"
  fi
  unzip -n -q "${DESTINATION}/${name}" -d "${DESTINATION}"
}

download cvprw26_trainval_instance_labels.zip
echo "b8b7e8202856966d0f4ff0e3b3aa9b77  ${DESTINATION}/cvprw26_trainval_instance_labels.zip" | md5sum --check
if [[ "${MODE}" == "full" ]]; then
  download pre-event.zip
  download post-event.zip
  download target.zip
  echo "087db04490233e40fd5b53ea1d3b374a  ${DESTINATION}/pre-event.zip" | md5sum --check
  echo "13dbfff273e95995fee2a868388da4ea  ${DESTINATION}/post-event.zip" | md5sum --check
  echo "d7f48f686e0b01772949c1b5e56e3146  ${DESTINATION}/target.zip" | md5sum --check
elif [[ "${MODE}" != "labels" ]]; then
  echo "usage: $0 [labels|full] [destination]" >&2
  exit 2
fi
