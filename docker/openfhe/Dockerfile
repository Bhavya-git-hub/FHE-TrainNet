# OpenFHE, for the one thing TenSEAL cannot do: actual CKKS bootstrapping.
#
# The `openfhe` wheel on PyPI is published for Linux and macOS only - its
# classifiers say so, and its description says "This release requires Ubuntu
# 24.04". It does not install on Windows, which is why this backend lives in a
# container rather than alongside the default one.
FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# libgomp1 is the GNU OpenMP runtime. OpenFHE is built against it and parallelises
# its lattice arithmetic with it, so without this package `import openfhe` fails
# with `ImportError: libgomp.so.1: cannot open shared object file` - and it fails
# at import, before any API exists to inspect, which is easy to misread as the
# library lacking a feature rather than not loading at all.
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
      python3 python3-pip python3-venv libgomp1 \
 && rm -rf /var/lib/apt/lists/*

# A virtualenv, because Ubuntu 24.04 marks the system Python as externally managed.
RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

RUN pip install --no-cache-dir "openfhe==1.5.1.0.24.4" "numpy>=1.26"

# The repository is mounted at /work at run time rather than copied in, so the
# refresh policies the container executes are the *same source files* the native
# backend uses. That is the point of this backend: only the primitive differs.
WORKDIR /work
ENV PYTHONPATH=/work

ENTRYPOINT ["python3"]
