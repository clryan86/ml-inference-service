# Security Policy

This repository uses bounded Pydantic feature validation and performs no arbitrary code execution from requests. Model artifacts are generated locally from repository code rather than downloaded from untrusted URLs.

Production hardening should add authentication/authorization, rate limits, request-size limits, signed/verified artifacts, private model storage, TLS, structured logging, abuse monitoring, and controls around any future retraining endpoint.
