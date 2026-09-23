#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
BASE_SPEC="${SCRIPT_DIR}/rf_diffusion_aa.spec"
DEFAULT_IMAGE_NAME="rf_diffusion_aa.sif"
DEFAULT_OUTPUT="${SCRIPT_DIR}/${DEFAULT_IMAGE_NAME}"

REPO_URL="https://github.com/RosettaCommons/RFDiffusion2_all_the_code.git"
REPO_CONTAINER_DIR="/opt/RFDiffusion2_all_the_code"
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

OUTPUT="${DEFAULT_OUTPUT}"
OUTPUT_EXPLICIT=0
IMAGE_NAME="${DEFAULT_IMAGE_NAME}"
WITH_REPO=0
REPO_REF="main"
WITH_WEIGHTS=0
KEEP_BUILD_DIR=0
BUILD_DIR=""

usage() {
    cat <<EOF
Usage: $(basename "$0") [options]

Build RFdiffusion-AA from ${BASE_SPEC}.

Options:
  --name NAME         Set the Apptainer image filename in ${SCRIPT_DIR}.
                      Defaults to ${DEFAULT_IMAGE_NAME}.
  --output PATH       Write the SIF to PATH. Defaults to ${DEFAULT_OUTPUT}.
                      Overrides --name.
  --with-repo         Clone ${REPO_URL} on the host and embed it at ${REPO_CONTAINER_DIR}.
  --repo-ref REF      Git ref to check out when --with-repo is used. Defaults to main.
  --with-weights      Download and embed all published model weights.
  --keep-build-dir    Keep the temporary derived spec and payload directory.
  -h, --help          Show this help message.

The build command uses apptainer when available, otherwise singularity:
  build --nv --notest OUTPUT SPEC
EOF
}

info() {
    printf '%s\n' "$*"
}

die() {
    printf 'Error: %s\n' "$*" >&2
    exit 2
}

cleanup() {
    if [[ -n "${BUILD_DIR}" && "${KEEP_BUILD_DIR}" -eq 0 ]]; then
        rm -rf -- "${BUILD_DIR}"
    fi
}
trap cleanup EXIT

download_file() {
    local url="$1"
    local output="$2"
    local partial="${output}.partial"

    rm -f -- "${partial}"
    if command -v curl >/dev/null 2>&1; then
        curl -fL --retry 3 --retry-delay 5 -o "${partial}" "${url}"
    elif command -v wget >/dev/null 2>&1; then
        wget -O "${partial}" "${url}"
    else
        die "curl or wget is required to download model weights"
    fi
    mv -- "${partial}" "${output}"
}

while [[ "$#" -gt 0 ]]; do
    case "$1" in
        --name)
            [[ "$#" -ge 2 && -n "$2" ]] || die "--name requires an image filename"
            IMAGE_NAME="$2"
            shift 2
            ;;
        --name=*)
            IMAGE_NAME="${1#--name=}"
            [[ -n "${IMAGE_NAME}" ]] || die "--name requires an image filename"
            shift
            ;;
        --output)
            [[ "$#" -ge 2 && -n "$2" ]] || die "--output requires a path"
            OUTPUT="$2"
            OUTPUT_EXPLICIT=1
            shift 2
            ;;
        --output=*)
            OUTPUT="${1#--output=}"
            [[ -n "${OUTPUT}" ]] || die "--output requires a path"
            OUTPUT_EXPLICIT=1
            shift
            ;;
        --with-repo)
            WITH_REPO=1
            shift
            ;;
        --repo-ref)
            [[ "$#" -ge 2 && -n "$2" ]] || die "--repo-ref requires a ref"
            REPO_REF="$2"
            shift 2
            ;;
        --repo-ref=*)
            REPO_REF="${1#--repo-ref=}"
            [[ -n "${REPO_REF}" ]] || die "--repo-ref requires a ref"
            shift
            ;;
        --with-weights)
            WITH_WEIGHTS=1
            shift
            ;;
        --keep-build-dir)
            KEEP_BUILD_DIR=1
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        --)
            shift
            break
            ;;
        *)
            die "unknown argument: $1"
            ;;
    esac
done

[[ "$#" -eq 0 ]] || die "unexpected positional argument: $1"
[[ -f "${BASE_SPEC}" ]] || die "base spec not found: ${BASE_SPEC}"

if [[ "${IMAGE_NAME}" == */* ]]; then
    die "--name must be a filename, not a path; use --output for custom paths"
fi
if [[ "${IMAGE_NAME}" != *.sif ]]; then
    IMAGE_NAME="${IMAGE_NAME}.sif"
fi
if [[ "${OUTPUT_EXPLICIT}" -eq 0 ]]; then
    OUTPUT="${SCRIPT_DIR}/${IMAGE_NAME}"
fi

if command -v apptainer >/dev/null 2>&1; then
    CONTAINER_BUILDER="$(command -v apptainer)"
elif command -v singularity >/dev/null 2>&1; then
    CONTAINER_BUILDER="$(command -v singularity)"
else
    die "apptainer or singularity must be available on PATH"
fi

SPEC_TO_BUILD="${BASE_SPEC}"

if [[ "${WITH_REPO}" -eq 1 || "${WITH_WEIGHTS}" -eq 1 ]]; then
    BUILD_DIR="$(mktemp -d "${TMPDIR:-/tmp}/rf_diffusion_aa_build.XXXXXX")"
    PAYLOAD_REPO_DIR="${BUILD_DIR}/payload/RFDiffusion2_all_the_code"

    if [[ "${WITH_REPO}" -eq 1 ]]; then
        info "Cloning ${REPO_URL} at ${REPO_REF}"
        mkdir -p -- "$(dirname -- "${PAYLOAD_REPO_DIR}")"
        git clone "${REPO_URL}" "${PAYLOAD_REPO_DIR}"
        git -C "${PAYLOAD_REPO_DIR}" checkout "${REPO_REF}"
        git -C "${PAYLOAD_REPO_DIR}" submodule update --init --recursive
    else
        mkdir -p -- "${PAYLOAD_REPO_DIR}"
    fi

    if [[ "${WITH_WEIGHTS}" -eq 1 ]]; then
        WEIGHTS_DIR="${PAYLOAD_REPO_DIR}/rf_diffusion/model_weights"
        mkdir -p -- "${WEIGHTS_DIR}"

        for index in "${!WEIGHT_NAMES[@]}"; do
            weight="${WEIGHT_NAMES[index]}"
            info "Downloading ${weight}"
            download_file "${WEIGHT_URLS[index]}" "${WEIGHTS_DIR}/${weight}"
        done
    fi

    FILES_SNIPPET="${BUILD_DIR}/files.snippet"
    ENV_SNIPPET="${BUILD_DIR}/environment.snippet"
    SPEC_TO_BUILD="${BUILD_DIR}/rf_diffusion_aa.with_payload.spec"

    {
        printf '%%files\n'
        printf '%s %s\n' "${PAYLOAD_REPO_DIR}" "${REPO_CONTAINER_DIR}"
    } >"${FILES_SNIPPET}"

    {
        printf '# Added by %s\n' "$(basename "$0")"
        printf 'export RFDIFFUSION2_REPO=%s\n' "${REPO_CONTAINER_DIR}"
        printf 'export RFDIFFUSION2_WEIGHTS_DIR=%s/rf_diffusion/model_weights\n' "${REPO_CONTAINER_DIR}"
        if [[ "${WITH_REPO}" -eq 1 ]]; then
            printf 'export PYTHONPATH=%s:$PYTHONPATH\n' "${REPO_CONTAINER_DIR}"
        fi
    } >"${ENV_SNIPPET}"

    awk -v files_snippet="${FILES_SNIPPET}" -v env_snippet="${ENV_SNIPPET}" '
function dump_file(path, line) {
    while ((getline line < path) > 0) {
        print line
    }
    close(path)
}

/^%post([[:space:]]|$)/ && !files_done {
    dump_file(files_snippet)
    print ""
    files_done = 1
}

/^%[[:alpha:]]/ && in_environment && $0 !~ /^%environment([[:space:]]|$)/ && !environment_done {
    dump_file(env_snippet)
    print ""
    environment_done = 1
    in_environment = 0
}

{
    print
}

/^%environment([[:space:]]|$)/ {
    in_environment = 1
}

END {
    if (!files_done) {
        print ""
        dump_file(files_snippet)
    }
    if (!environment_done) {
        if (!in_environment) {
            print ""
            print "%environment"
        }
        dump_file(env_snippet)
    }
}
' "${BASE_SPEC}" >"${SPEC_TO_BUILD}"

    info "Generated derived spec at ${SPEC_TO_BUILD}"
fi

mkdir -p -- "$(dirname -- "${OUTPUT}")"
info "Building ${OUTPUT}"
"${CONTAINER_BUILDER}" build --nv --notest "${OUTPUT}" "${SPEC_TO_BUILD}"

if [[ -n "${BUILD_DIR}" && "${KEEP_BUILD_DIR}" -eq 1 ]]; then
    info "Kept build directory: ${BUILD_DIR}"
fi
