#!/usr/bin/env bash
# Copy the course-internal Booster snapshot without publishing it in this repo.

set -euo pipefail

COURSE_ROOT="${1:-../course}"
SOURCE="${COURSE_ROOT}/hw3/vendor/booster_mjlab"
TARGET="vendor/booster_mjlab"

if [[ ! -f "${SOURCE}/pyproject.toml" ]]; then
  echo "ERROR: Booster course snapshot not found at ${SOURCE}" >&2
  echo "Usage: $0 /path/to/course" >&2
  exit 1
fi

if [[ -e "${TARGET}" ]]; then
  echo "Vendor already exists at ${TARGET}; leaving it unchanged."
  exit 0
fi

mkdir -p vendor
cp -a "${SOURCE}" "${TARGET}"
echo "Copied course Booster snapshot to ${TARGET}."
