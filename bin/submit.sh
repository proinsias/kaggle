#!/usr/bin/env bash

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

# shellcheck source=bin/common.sh
source "${SCRIPT_DIR}"/common.sh

kaggle competitions submit -c "${COMPETITION}" -f "${1}" -m "${2}"
