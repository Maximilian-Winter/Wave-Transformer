"""
Normalization specification system for signal transformers.

Provides flexible, serializable, and differentiable normalization options
that can be applied after activation functions in signal encoders.
"""

from enum import Enum
from dataclasses import dataclass, field
from typing import Callable, Optional, Union, Dict, Any
import torch
import numpy as np
import ast
import operator


class NormalizationType(Enum):
    """Predefined normalization schemes."""

    # Identity (no normalization)
    IDENTITY = "identity"

    # Linear scaling: a*x + b
    LINEAR = "linear"

    # Bounded ranges
    TANH_SCALED = "tanh_scaled"  # tanh(x) * scale + offset
    SIGMOID_SCALED = "sigmoid_scaled"  # sigmoid(x) * scale + offset

    # Clipping
    CLIP = "clip"  # torch.clamp(x, min, max)

    # Periodic/angular
    ANGULAR = "angular"  # x * pi (for angular features)

    # Custom expression
    EXPRESSION = "expression"  # Parse string expression

    # Custom callable
    CUSTOM = "custom"  # User-defined function


@dataclass
class NormalizationParams:
    """
    Parameters for normalization functions.
    All parameters are optional and depend on the normalization type.
    """
    # Linear scaling parameters
    scale: float = 1.0
    offset: float = 0.0

    # Clipping parameters
    min_val: Optional[float] = None
    max_val: Optional[float] = None

    # Expression string (for EXPRESSION type)
    expression: Optional[str] = None

    # Function name for serialization (for CUSTOM type)
    function_name: Optional[str] = None

    # Additional parameters as dict for extensibility
    extra: Dict[str, Any] = field(default_factory=dict)


class ExpressionParser:
    """
    Safe parser for mathematical expressions involving 'x'.
    Supports: +, -, *, /, **, numpy constants (pi, e), torch functions.
    """

    # Allowed operations
    OPERATORS = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.Pow: operator.pow,
        ast.USub: operator.neg,
    }

    # Allowed constants
    CONSTANTS = {
        'pi': np.pi,
        'e': np.e,
        'inf': float('inf'),
    }

    # Allowed torch functions
    TORCH_FUNCTIONS = {
        'tanh': torch.tanh,
        'sigmoid': torch.sigmoid,
        'relu': torch.relu,
        'abs': torch.abs,
        'exp': torch.exp,
        'log': torch.log,
        'sqrt': torch.sqrt,
        'sin': torch.sin,
        'cos': torch.cos,
        'tan': torch.tan,
    }

    def __init__(self, expression: str):
        """
        Initialize parser with an expression string.

        Args:
            expression: Mathematical expression string (e.g., "x * 2.0 + 1.0", "tanh(x) * pi")
        """
        self.expression = expression.strip()
        self._validate_expression()

    def _validate_expression(self):
        """Validate that expression is safe to parse."""
        # Check for dangerous patterns
        dangerous_patterns = [
            r'__', 'import', 'eval', 'exec', 'open', 'file',
            'compile', 'globals', 'locals', 'vars'
        ]

        for pattern in dangerous_patterns:
            if pattern in self.expression:
                raise ValueError(f"Forbidden pattern '{pattern}' in expression: {self.expression}")

    def _eval_node(self, node, x: torch.Tensor):
        """Recursively evaluate AST node."""
        if isinstance(node, ast.Constant):  # Python 3.8+
            return node.value
        elif isinstance(node, ast.Num):  # Python 3.7
            return node.n
        elif isinstance(node, ast.Name):
            if node.id == 'x':
                return x
            elif node.id in self.CONSTANTS:
                return self.CONSTANTS[node.id]
            else:
                raise ValueError(f"Unknown variable: {node.id}")
        elif isinstance(node, ast.BinOp):
            left = self._eval_node(node.left, x)
            right = self._eval_node(node.right, x)
            op = self.OPERATORS.get(type(node.op))
            if op is None:
                raise ValueError(f"Unsupported operation: {type(node.op)}")
            return op(left, right)
        elif isinstance(node, ast.UnaryOp):
            operand = self._eval_node(node.operand, x)
            op = self.OPERATORS.get(type(node.op))
            if op is None:
                raise ValueError(f"Unsupported unary operation: {type(node.op)}")
            return op(operand)
        elif isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name):
                raise ValueError("Only named function calls are supported")

            func_name = node.func.id
            if func_name not in self.TORCH_FUNCTIONS:
                raise ValueError(f"Unknown function: {func_name}")

            func = self.TORCH_FUNCTIONS[func_name]

            # Evaluate arguments
            args = [self._eval_node(arg, x) for arg in node.args]

            return func(*args)
        else:
            raise ValueError(f"Unsupported AST node: {type(node)}")

    def parse(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parse and evaluate expression with input tensor.

        Args:
            x: Input tensor

        Returns:
            Transformed tensor
        """
        try:
            tree = ast.parse(self.expression, mode='eval')
            result = self._eval_node(tree.body, x)

            # Convert to tensor if needed
            if not isinstance(result, torch.Tensor):
                result = torch.tensor(result, dtype=x.dtype, device=x.device)

            return result
        except Exception as e:
            raise ValueError(f"Failed to parse expression '{self.expression}': {str(e)}")


class NormalizationSpec:
    """
    Complete normalization specification that can be serialized and applied.
    """

    def __init__(
        self,
        norm_type: Union[NormalizationType, str],
        params: Optional[NormalizationParams] = None,
        custom_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None
    ):
        """
        Initialize normalization specification.

        Args:
            norm_type: Type of normalization (enum or string)
            params: Parameters for normalization
            custom_fn: Custom callable function (only for CUSTOM type)
        """
        if isinstance(norm_type, str):
            norm_type = NormalizationType(norm_type)

        self.norm_type = norm_type
        self.params = params or NormalizationParams()
        self.custom_fn = custom_fn

        # Validate configuration
        self._validate()

        # Cache parsed expression if needed
        self._expression_parser = None
        if self.norm_type == NormalizationType.EXPRESSION:
            if not self.params.expression:
                raise ValueError("EXPRESSION type requires 'expression' parameter")
            self._expression_parser = ExpressionParser(self.params.expression)

    def _validate(self):
        """Validate normalization configuration."""
        if self.norm_type == NormalizationType.CUSTOM:
            if self.custom_fn is None:
                raise ValueError("CUSTOM type requires custom_fn")
            if not callable(self.custom_fn):
                raise ValueError("custom_fn must be callable")

        if self.norm_type == NormalizationType.EXPRESSION:
            if not self.params.expression:
                raise ValueError("EXPRESSION type requires expression parameter")

        if self.norm_type == NormalizationType.CLIP:
            if self.params.min_val is None and self.params.max_val is None:
                raise ValueError("CLIP type requires at least min_val or max_val")

    def apply(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply normalization to input tensor.

        Args:
            x: Input tensor of any shape

        Returns:
            Normalized tensor of same shape
        """
        if self.norm_type == NormalizationType.IDENTITY:
            return x

        elif self.norm_type == NormalizationType.LINEAR:
            return x * self.params.scale + self.params.offset

        elif self.norm_type == NormalizationType.TANH_SCALED:
            return torch.tanh(x) * self.params.scale + self.params.offset

        elif self.norm_type == NormalizationType.SIGMOID_SCALED:
            return torch.sigmoid(x) * self.params.scale + self.params.offset

        elif self.norm_type == NormalizationType.CLIP:
            return torch.clamp(x, min=self.params.min_val, max=self.params.max_val)

        elif self.norm_type == NormalizationType.ANGULAR:
            return x * np.pi

        elif self.norm_type == NormalizationType.EXPRESSION:
            return self._expression_parser.parse(x)

        elif self.norm_type == NormalizationType.CUSTOM:
            return self.custom_fn(x)

        else:
            raise ValueError(f"Unknown normalization type: {self.norm_type}")

    def to_dict(self) -> Dict[str, Any]:
        """
        Serialize to dictionary for config saving.

        Returns:
            Dictionary representation
        """
        config = {
            'type': self.norm_type.value,
            'params': {
                'scale': self.params.scale,
                'offset': self.params.offset,
            }
        }

        # Add optional parameters if set
        if self.params.min_val is not None:
            config['params']['min_val'] = self.params.min_val
        if self.params.max_val is not None:
            config['params']['max_val'] = self.params.max_val
        if self.params.expression:
            config['params']['expression'] = self.params.expression
        if self.params.function_name:
            config['params']['function_name'] = self.params.function_name
        if self.params.extra:
            config['params']['extra'] = self.params.extra

        return config

    @classmethod
    def from_dict(cls, config: Dict[str, Any], custom_fn_registry: Optional[Dict[str, Callable]] = None) -> 'NormalizationSpec':
        """
        Deserialize from dictionary.

        Args:
            config: Dictionary configuration
            custom_fn_registry: Optional registry of custom functions by name

        Returns:
            NormalizationSpec instance
        """
        norm_type = NormalizationType(config['type'])
        params_dict = config.get('params', {})

        # Extract extra params separately
        extra = params_dict.pop('extra', {})

        params = NormalizationParams(
            scale=params_dict.get('scale', 1.0),
            offset=params_dict.get('offset', 0.0),
            min_val=params_dict.get('min_val'),
            max_val=params_dict.get('max_val'),
            expression=params_dict.get('expression'),
            function_name=params_dict.get('function_name'),
            extra=extra
        )

        # Handle custom functions
        custom_fn = None
        if norm_type == NormalizationType.CUSTOM:
            if params.function_name and custom_fn_registry:
                custom_fn = custom_fn_registry.get(params.function_name)
                if custom_fn is None:
                    raise ValueError(f"Custom function '{params.function_name}' not found in registry")
            else:
                raise ValueError("CUSTOM type requires function_name and custom_fn_registry")

        return cls(norm_type, params, custom_fn)

    def __repr__(self) -> str:
        if self.norm_type == NormalizationType.EXPRESSION:
            return f"NormalizationSpec(EXPRESSION: '{self.params.expression}')"
        elif self.norm_type == NormalizationType.CUSTOM:
            return f"NormalizationSpec(CUSTOM: {self.params.function_name or 'unnamed'})"
        else:
            return f"NormalizationSpec({self.norm_type.value}, scale={self.params.scale}, offset={self.params.offset})"


# ============= Convenience Constructors =============

def identity_norm() -> NormalizationSpec:
    """No normalization."""
    return NormalizationSpec(NormalizationType.IDENTITY)


def linear_norm(scale: float = 1.0, offset: float = 0.0) -> NormalizationSpec:
    """Linear normalization: x * scale + offset"""
    return NormalizationSpec(
        NormalizationType.LINEAR,
        NormalizationParams(scale=scale, offset=offset)
    )


def tanh_scaled(scale: float = 1.0, offset: float = 0.0) -> NormalizationSpec:
    """Tanh normalization: tanh(x) * scale + offset"""
    return NormalizationSpec(
        NormalizationType.TANH_SCALED,
        NormalizationParams(scale=scale, offset=offset)
    )


def sigmoid_scaled(scale: float = 1.0, offset: float = 0.0) -> NormalizationSpec:
    """Sigmoid normalization: sigmoid(x) * scale + offset"""
    return NormalizationSpec(
        NormalizationType.SIGMOID_SCALED,
        NormalizationParams(scale=scale, offset=offset)
    )


def clip_norm(min_val: Optional[float] = None, max_val: Optional[float] = None) -> NormalizationSpec:
    """Clip normalization: clamp(x, min, max)"""
    return NormalizationSpec(
        NormalizationType.CLIP,
        NormalizationParams(min_val=min_val, max_val=max_val)
    )


def angular_norm() -> NormalizationSpec:
    """Angular normalization: x * pi"""
    return NormalizationSpec(NormalizationType.ANGULAR)


def expression_norm(expression: str) -> NormalizationSpec:
    """Expression-based normalization from string."""
    return NormalizationSpec(
        NormalizationType.EXPRESSION,
        NormalizationParams(expression=expression)
    )


def custom_norm(fn: Callable, name: Optional[str] = None) -> NormalizationSpec:
    """Custom callable normalization."""
    return NormalizationSpec(
        NormalizationType.CUSTOM,
        NormalizationParams(function_name=name),
        custom_fn=fn
    )
