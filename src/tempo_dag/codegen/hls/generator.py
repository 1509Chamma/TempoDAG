from __future__ import annotations

from pathlib import Path
from string import Template
from typing import TYPE_CHECKING

from tempo_dag.ir.op import Operator, OperatorError

if TYPE_CHECKING:
    from collections.abc import Mapping

    from tempo_dag.ir.value import Value


PROJECT_ROOT = Path(__file__).resolve().parents[4]


class HLSRenderError(OperatorError):
    """Base exception for HLS template resolution and rendering failures."""


class HLSTemplateNotFoundError(HLSRenderError):
    """Raised when an operator references a missing HLS template file."""


class HLSTemplateRenderError(HLSRenderError):
    """Raised when a template cannot be rendered with the provided context."""


def resolve_hls_template_path(operator: Operator) -> Path:
    """Resolve an operator template path against the repo or module location."""

    raw_path = Path(operator.hls_template_path())
    if raw_path.is_absolute():
        if raw_path.is_file():
            return raw_path
        raise HLSTemplateNotFoundError(
            f"HLS template '{raw_path}' does not exist for operator '{operator.op_id}'"
        )

    project_candidate = (PROJECT_ROOT / raw_path).resolve()
    if project_candidate.is_file():
        return project_candidate

    module_file = _operator_module_file(operator)
    if module_file is not None:
        module_candidate = (module_file.parent / raw_path).resolve()
        if module_candidate.is_file():
            return module_candidate

    raise HLSTemplateNotFoundError(
        f"HLS template '{raw_path}' could not be resolved "
        f"for operator '{operator.op_id}'"
    )


def render_operator_hls(operator: Operator, values: Mapping[str, Value]) -> str:
    """Validate an operator and render its HLS template with `string.Template`."""

    operator.validate(values)
    template_path = resolve_hls_template_path(operator)
    template = Template(template_path.read_text(encoding="utf-8"))

    try:
        return template.substitute(operator.hls_context(values))
    except KeyError as exc:
        missing_key = exc.args[0]
        raise HLSTemplateRenderError(
            f"HLS template '{template_path}' is missing context value "
            f"'{missing_key}' for operator '{operator.op_id}'"
        ) from exc
    except ValueError as exc:
        raise HLSTemplateRenderError(
            f"HLS template '{template_path}' is invalid for operator '{operator.op_id}'"
        ) from exc


def _operator_module_file(operator: Operator) -> Path | None:
    module = __import__(operator.__class__.__module__, fromlist=["__file__"])
    module_file = getattr(module, "__file__", None)
    if module_file is None:
        return None
    return Path(module_file).resolve()

