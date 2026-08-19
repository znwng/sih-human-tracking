#!/usr/bin/env bash

set -euo pipefail

IMAGE="zenwing/human-tracking-model-devenv:latest"

usage() {
	echo "Usage: ./dev.sh {build|rebuild|run}"
	exit 1
}

build() {
	echo "==> Building $IMAGE..."
	docker build -t "$IMAGE" .
}

rebuild() {
	echo "==> Rebuilding $IMAGE from scratch..."
	docker build --no-cache -t "$IMAGE" .
}

run() {
	echo "==> Starting development container..."

	DOCKER_ARGS=(
		--rm
		-it
		-v "$(pwd):/workspace"
	)

	# Enable OpenCV GUI support on Linux/Ubuntu
	if [[ "$(uname -s)" == "Linux" ]] && [[ -n "${DISPLAY:-}" ]]; then
		echo "==> Linux detected: enabling GUI support"

		xhost +local:docker >/dev/null

		DOCKER_ARGS+=(
			-e "DISPLAY=$DISPLAY"
			-v "/tmp/.X11-unix:/tmp/.X11-unix:rw"
		)
	fi

	docker run "${DOCKER_ARGS[@]}" "$IMAGE"
}

if [[ $# -ne 1 ]]; then
	usage
fi

case "$1" in
build)
	build
	;;
rebuild)
	rebuild
	;;
run)
	run
	;;
*)
	usage
	;;
esac
