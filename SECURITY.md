# Security Policy

TempoDAG is research-stage compiler infrastructure. It is not currently
intended for production deployment in safety-critical or financially regulated
systems.

## Supported Versions

Security fixes are applied to the `main` branch. There are no released stable
versions yet.

## Reporting A Vulnerability

Please report suspected vulnerabilities privately instead of opening a public
issue. Use GitHub's private vulnerability reporting if it is enabled for the
repository, or contact the maintainer listed on the repository profile.

Useful reports include:

- Affected file, workflow, parser, or generated artifact.
- Reproduction steps.
- Expected impact.
- Suggested mitigation, if known.

## Scope

Security-sensitive areas include:

- Parsing untrusted ONNX, TensorFlow, or PyTorch-exported models.
- Generated C++/HLS code and script emission.
- GitHub Actions workflows and dependency automation.
- Future board, RTL, or toolchain integration scripts.

Do not run untrusted models, generated code, or vendor-tool scripts without
reviewing them first.
