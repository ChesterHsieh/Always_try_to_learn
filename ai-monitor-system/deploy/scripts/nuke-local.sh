#!/usr/bin/env bash
# Dev-only brute-force teardown: wipes the ai-monitor-system namespace
# plus any PVs/PVCs left behind. Not for any shared or production cluster.
set -euo pipefail

NAMESPACE="${NAMESPACE:-ai-monitor-system}"
RELEASE_NAME="${RELEASE_NAME:-monitor}"

echo "[nuke] Uninstalling Helm release ${RELEASE_NAME} (if present)"
helm uninstall "${RELEASE_NAME}" -n "${NAMESPACE}" >/dev/null 2>&1 || true

echo "[nuke] Deleting namespace ${NAMESPACE}"
kubectl delete namespace "${NAMESPACE}" --wait=false --ignore-not-found

echo "[nuke] Force-finalizing stuck namespace (if any)"
if kubectl get namespace "${NAMESPACE}" >/dev/null 2>&1; then
  kubectl get namespace "${NAMESPACE}" -o json \
    | sed 's/"kubernetes"//g' \
    | kubectl replace --raw "/api/v1/namespaces/${NAMESPACE}/finalize" -f - >/dev/null 2>&1 || true
fi

echo "[nuke] Deleting released PVs that belonged to ${NAMESPACE}"
kubectl get pv -o json \
  | jq -r --arg ns "${NAMESPACE}" \
      '.items[] | select(.spec.claimRef.namespace==$ns) | .metadata.name' \
  | xargs -r -n1 kubectl delete pv --ignore-not-found

echo "[nuke] Done."