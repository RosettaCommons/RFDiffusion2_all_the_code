#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
SIF_PATH="${REPO_DIR}/rf_diffusion/exec/rf_diffusion_aa.sif"
TMP_DIR="${REPO_DIR}/.apptainer-tmp"
URI="oras://docker.io/magnusbauer/rfdiffusion2-apptainer:portable"
SHA256="f8bdfd4e9570fe4091931512a2570b71729a110efdb7b908c7f2c67cfbb9b025"
WEIGHT_NAMES=(
    RFD_173.pt
    RFD_140.pt
    RFD_45.pt
    ppi_robust_struct.pt
)
WEIGHT_URLS=(
    https://files.ipd.uw.edu/pub/rfdiffusion2/model_weights/RFD_173.pt
    https://files.ipd.uw.edu/pub/rfdiffusion2/model_weights/RFD_140.pt
    https://files.ipd.uw.edu/pub/rfdiffusion2-mi/model_weights/RFD_45.pt
    https://files.ipd.uw.edu/pub/rfdiffusion2-mi/model_weights/ppi_robust_struct.pt
)

usage() {
    cat <<EOF
Usage: $(basename "$0") MODE [options]

Modes:
  weights             Download model weights.
  apptainer           Pull and verify the portable Apptainer image.
  all                 Run both modes.

Options:
  --output-dir DIR    Weight output directory.
  --output PATH       Apptainer SIF output path.
  --tmp-dir DIR       Apptainer temporary directory.
  --force             Replace existing SIF or weight files.
  --keep-temp         Keep Apptainer temporary files.
  --dry-run           Print actions without downloading.
  -h, --help          Show this help.
EOF
}

die() { printf 'Error: %s\n' "$*" >&2; exit 2; }
[[ "$#" -gt 0 ]] || { usage; exit 2; }
MODE="$1"; shift
case "${MODE}" in weights|apptainer|all) ;; -h|--help) usage; exit 0 ;; *) die "unknown mode: ${MODE}" ;; esac

WEIGHTS_DIR="${REPO_DIR}/rf_diffusion/model_weights"
FORCE=0
DRY_RUN=0
KEEP_TEMP=0
while [[ "$#" -gt 0 ]]; do
    case "$1" in
        --output-dir) [[ "$#" -ge 2 ]] || die "--output-dir requires a directory"; WEIGHTS_DIR="$2"; shift 2 ;;
        --output-dir=*) WEIGHTS_DIR="${1#--output-dir=}"; shift ;;
        --output) [[ "$#" -ge 2 ]] || die "--output requires a path"; SIF_PATH="$2"; shift 2 ;;
        --output=*) SIF_PATH="${1#--output=}"; shift ;;
        --tmp-dir) [[ "$#" -ge 2 ]] || die "--tmp-dir requires a directory"; TMP_DIR="$2"; shift 2 ;;
        --tmp-dir=*) TMP_DIR="${1#--tmp-dir=}"; shift ;;
        --force) FORCE=1; shift ;;
        --keep-temp) KEEP_TEMP=1; shift ;;
        --dry-run) DRY_RUN=1; shift ;;
        -h|--help) usage; exit 0 ;;
        *) die "unknown argument: $1" ;;
    esac
done

download_apptainer() {
    local actual staging_dir
    command -v apptainer >/dev/null 2>&1 || die "apptainer is required"
    command -v sha256sum >/dev/null 2>&1 || die "sha256sum is required"

    if [[ -f "${SIF_PATH}" && "${FORCE}" -eq 0 ]]; then
        actual="$(sha256sum -- "${SIF_PATH}" | awk '{print $1}')"
        if [[ "${actual}" == "${SHA256}" ]]; then
            printf 'Already present and verified, skipping: %s\n' "${SIF_PATH}"
            return
        fi
        die "existing SIF has checksum ${actual}; use --force to replace it"
    fi
    if [[ "${DRY_RUN}" -eq 1 ]]; then
        printf 'apptainer pull --disable-cache %s <- %s\n' "${SIF_PATH}" "${URI}"
        return
    fi

    mkdir -p -- "$(dirname -- "${SIF_PATH}")" "${TMP_DIR}"
    staging_dir="$(mktemp -d "${TMP_DIR%/}/pull.XXXXXX")"
    if [[ "${KEEP_TEMP}" -eq 0 ]]; then trap 'rm -rf -- "${staging_dir}"' EXIT; fi
    if [[ "${FORCE}" -eq 1 ]]; then
        TMPDIR="${staging_dir}" APPTAINER_TMPDIR="${staging_dir}" \
            apptainer pull --force --disable-cache "${SIF_PATH}" "${URI}"
    else
        TMPDIR="${staging_dir}" APPTAINER_TMPDIR="${staging_dir}" \
            apptainer pull --disable-cache "${SIF_PATH}" "${URI}"
    fi
    actual="$(sha256sum -- "${SIF_PATH}" | awk '{print $1}')"
    [[ "${actual}" == "${SHA256}" ]] || die "SIF checksum mismatch: expected ${SHA256}, got ${actual}"
    printf 'Verified Apptainer image: %s\n' "${actual}"
    if [[ "${KEEP_TEMP}" -eq 0 ]]; then
        rm -rf -- "${staging_dir}"
        trap - EXIT
    fi
}

download_weights() {
    local downloader name url output partial index

    if command -v curl >/dev/null 2>&1; then
        downloader=curl
    elif command -v wget >/dev/null 2>&1; then
        downloader=wget
    else
        die "curl or wget is required"
    fi

    [[ "${DRY_RUN}" -eq 1 ]] || mkdir -p -- "${WEIGHTS_DIR}"
    for index in "${!WEIGHT_NAMES[@]}"; do
        name="${WEIGHT_NAMES[index]}"
        url="${WEIGHT_URLS[index]}"
        output="${WEIGHTS_DIR}/${name}"
        partial="${output}.partial"

        if [[ -f "${output}" && "${FORCE}" -eq 0 ]]; then
            printf 'Already present, skipping: %s\n' "${output}"
            continue
        fi
        if [[ "${DRY_RUN}" -eq 1 ]]; then
            printf '%s -> %s\n' "${url}" "${output}"
            continue
        fi
        [[ "${FORCE}" -eq 0 ]] || rm -f -- "${partial}"
        printf 'Downloading %s\n' "${name}"
        if [[ "${downloader}" == curl ]]; then
            curl -fL --retry 3 --retry-delay 5 --continue-at - \
                -o "${partial}" "${url}"
        else
            wget --continue -O "${partial}" "${url}"
        fi
        mv -f -- "${partial}" "${output}"
    done
}

case "${MODE}" in
    weights) download_weights ;;
    apptainer) download_apptainer ;;
    all) download_apptainer; download_weights ;;
esac
