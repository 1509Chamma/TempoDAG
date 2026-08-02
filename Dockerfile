FROM python:3.12-slim

ENV DEBIAN_FRONTEND=noninteractive \
    MPLBACKEND=Agg \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Research/notebook tooling first, then the frozen runtime set so its pins
# win wherever the two overlap.
COPY research/requirements-research.txt /tmp/requirements-research.txt
COPY requirements.txt /tmp/requirements.txt
# The final uninstall: torch pulls triton on Linux; its wheel faults on
# CPU-only machines and nothing here uses torch.compile (CI does the same).
RUN python -m pip install --upgrade pip \
    && python -m pip install -r /tmp/requirements-research.txt \
    && python -m pip install -r /tmp/requirements.txt \
    && (python -m pip uninstall -y triton 2>/dev/null || true)

WORKDIR /repo
COPY . /repo

# Default: reproduce everything that needs only Python (tests, walkthroughs,
# research scripts, tutorial, HLS emission). The Vitis ladder is the one
# non-portable stage - see scripts/reproduce.sh.
ENTRYPOINT ["bash", "scripts/reproduce.sh"]
CMD ["all"]
