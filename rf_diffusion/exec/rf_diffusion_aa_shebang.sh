#!/usr/bin/bash

###################
# You can add the path to this file as the shebang line in your python script. 
# Then by default, the python script will be executed with the python interpreter
# in the SIF_PATH container. Here, we launch the container with nvidia gpu and slurm support.
#
# Example shebang: #!/usr/bin/env -S /bin/sh -c '"$(dirname "$0")/exec/rf_diffusion_aa_shebang.sh" "$0" "$@"'
###################

# Let the user know this script is setting things up behind the scene
SCRIPT_PATH=$(realpath "$0")
SCRIPT_DIR=$(dirname "$SCRIPT_PATH")
DEFAULT_SIF_PATH="$SCRIPT_DIR/rf_diffusion_aa.sif"
DEFAULT_SIF_URI="oras://docker.io/magnusbauer/rfdiffusion2-apptainer:portable"
SIF_PATH="${RFDIFFUSION2_SIF_PATH:-$DEFAULT_SIF_PATH}"
SIF_URI="${RFDIFFUSION2_APPTAINER_URI:-$DEFAULT_SIF_URI}"
echo '################## Start shebang info ##################'
echo "The file $SCRIPT_PATH is being run as a shebang executable. It will...
    1. Add the "rf_diffusion" repo directory to your PYTHONPATH.
    2. Run your python script from the right container, which contains all dependencies.
    3. Launch the container with slurm and nvidia gpu support."

# Extract the path to the Python script from the arguments
PYTHON_SCRIPT=$(realpath "$1")
shift

# Automatically add the git repo dir to the PYTHONPATH
PKG_NAME=rf_diffusion
if [[ $PYTHON_SCRIPT =~ $PKG_NAME ]]
then
    PKG_DIR=$(echo "$PYTHON_SCRIPT" | sed -E 's/^(.*\/'$PKG_NAME'\/).*/\1/')
    REPO_DIR=$(dirname "$PKG_DIR")
    
    if [[ $PYTHONPATH =~ $REPO_DIR ]]
    then
        echo "The repo dir ($REPO_DIR) is already in the PYTHONPATH. PYTHONPATH will remain as $PYTHONPATH"
    else
        export PYTHONPATH=$PYTHONPATH:$REPO_DIR
        echo "The repo dir ($REPO_DIR) was not in the PYTHONPATH. PYTHONPATH is now $PYTHONPATH"
    fi

else
    echo "The script $PYTHON_SCRIPT is not in the \"rf_diffusion\" package. Are you sure sure you're using the right shebang?"
    exit 1
fi

export RFDIFFUSION2_WEIGHTS_DIR="${RFDIFFUSION2_WEIGHTS_DIR:-$REPO_DIR/rf_diffusion/model_weights}"

if [ -n "${APPTAINER_NAME:-}" ]; then
    echo "Already running inside container $APPTAINER_NAME. Executing $PYTHON_SCRIPT with $(which python) in the existing container."
else
    if command -v apptainer >/dev/null 2>&1; then
        APPTAINER_BIN=$(command -v apptainer)
    elif command -v singularity >/dev/null 2>&1; then
        APPTAINER_BIN=$(command -v singularity)
    else
        echo "Apptainer (or Singularity) is required to run $PYTHON_SCRIPT." >&2
        echo "Install Apptainer, then run this script again:" >&2
        echo "https://apptainer.org/docs/admin/main/installation.html" >&2
        exit 127
    fi

    if [ ! -f "$SIF_PATH" ]; then
        echo "RFdiffusion2 Apptainer image not found: $SIF_PATH"
        echo "Prebuilt image: $SIF_URI"

        if [ -t 0 ]; then
            printf 'Download the prebuilt image from Docker Hub now (about 14 GB)? [y/N] '
            read -r DOWNLOAD_REPLY
        else
            DOWNLOAD_REPLY=""
        fi

        case "$DOWNLOAD_REPLY" in
            y|Y|yes|YES|Yes)
                mkdir -p "$(dirname "$SIF_PATH")"
                PARTIAL_SIF="${SIF_PATH}.partial.sif"
                trap 'rm -f -- "$PARTIAL_SIF"' EXIT
                "$APPTAINER_BIN" pull --force "$PARTIAL_SIF" "$SIF_URI"
                mv -- "$PARTIAL_SIF" "$SIF_PATH"
                trap - EXIT
                ;;
            *)
                echo "Image download was not started." >&2
                echo "Download it later with:" >&2
                echo "  $APPTAINER_BIN pull '$SIF_PATH' '$SIF_URI'" >&2
                echo "Or build it with:" >&2
                echo "  $SCRIPT_DIR/build_rf_diffusion_aa_apptainer.sh" >&2
                exit 2
                ;;
        esac
    fi

    echo "Running $PYTHON_SCRIPT with $SIF_PATH."
    echo '################## End shebang info ####################'
    echo
    "$APPTAINER_BIN" run --nv \
        --env PYTHONPATH="\$PYTHONPATH:$PYTHONPATH" \
        --env RFDIFFUSION2_WEIGHTS_DIR="$RFDIFFUSION2_WEIGHTS_DIR" \
        "$SIF_PATH" "$PYTHON_SCRIPT" "$@"
    exit $?
fi

echo '################## End shebang info ####################'
echo
python "$PYTHON_SCRIPT" "$@"
