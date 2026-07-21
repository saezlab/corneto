import ast
import re
from functools import lru_cache
from multiprocessing import Pool, cpu_count
from typing import Callable

from corneto._settings import LOGGER

_pattern = r"\b(?!and\b|or\b)[A-Za-z0-9_]+\b"

# Gene token regex presets.
# Keep "symbol_simple" equal to historical default for backward compatibility.
GENE_PATTERN_PRESETS = {
    "ecoli_bnumber": r"\bb\d{4}\b",
    "symbol_simple": _pattern,
    "symbol_extended": r"\b(?!and\b|or\b)[A-Za-z0-9_.:-]+\b",
    "uniprot_acc": r"\b(?:[OPQ][0-9][A-Z0-9]{3}[0-9]|[A-NR-Z][0-9][A-Z0-9]{3}[0-9])\b",
}
DEFAULT_PATTERN_MODE = "symbol_simple"


def resolve_pattern(pattern=None, pattern_mode=DEFAULT_PATTERN_MODE):
    if pattern is not None:
        return pattern
    if pattern_mode not in GENE_PATTERN_PRESETS:
        raise ValueError(
            f"Unknown pattern_mode={pattern_mode!r}. "
            f"Available: {sorted(GENE_PATTERN_PRESETS)}"
        )
    return GENE_PATTERN_PRESETS[pattern_mode]


def _and(a, b):
    if a < 0 and b < 0:
        return max(a, b)
    return min(a, b)


def _or(a, b):
    if a < 0 and b < 0:
        return min(a, b)
    return max(a, b)


def get_genes_from_gpr(
    gpr_expression, regex=_pattern, pattern_mode=DEFAULT_PATTERN_MODE
):
    if not isinstance(gpr_expression, str) or not gpr_expression.strip():
        return set()
    pattern = resolve_pattern(pattern=regex, pattern_mode=pattern_mode)
    return set(re.findall(pattern, gpr_expression))


def get_unique_genes(
    G,
    gpr_field="GPR",
    startswith=None,
    regex=_pattern,
    pattern_mode=DEFAULT_PATTERN_MODE,
):
    ugenes = set()
    for i in range(G.num_edges):
        gpr = G.get_attr_edge(i).get(gpr_field, "")
        genes = get_genes_from_gpr(gpr, regex=regex, pattern_mode=pattern_mode)
        if startswith is not None:
            genes = {g for g in genes if g.startswith(startswith)}
        ugenes |= genes
    return ugenes


@lru_cache(maxsize=20000)
def _parse_gpr_cached(expression):
    try:
        return ast.parse(expression, mode="eval")
    except SyntaxError:
        return None


def _eval_gpr(node, context, func_and, func_or, expression=None, default_value=0):
    if isinstance(node, ast.Expression):
        return _eval_gpr(
            node.body,
            context,
            func_and,
            func_or,
            expression=expression,
            default_value=default_value,
        )
    elif isinstance(node, ast.BoolOp):
        # Process the first value to initialize the result,
        # then iterate through the remaining values if any.
        values_iter = iter(node.values)
        result = _eval_gpr(
            next(values_iter),
            context,
            func_and,
            func_or,
            expression=expression,
            default_value=default_value,
        )
        for value in values_iter:
            if isinstance(node.op, ast.And):
                result = func_and(
                    result,
                    _eval_gpr(
                        value,
                        context,
                        func_and,
                        func_or,
                        expression=expression,
                        default_value=default_value,
                    ),
                )
            elif isinstance(node.op, ast.Or):
                result = func_or(
                    result,
                    _eval_gpr(
                        value,
                        context,
                        func_and,
                        func_or,
                        expression=expression,
                        default_value=default_value,
                    ),
                )
        return result
    elif isinstance(node, ast.Name):
        return context.get(node.id, default_value)
    elif isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)):
            return float(node.value)
        return default_value
    else:
        LOGGER.warning(
            f"Unsupported AST node: {type(node).__name__}, expression = {expression}"
        )
        return default_value


def evaluate_gpr(
    expression: str,
    symbol_values: dict,
    func_and=_and,
    func_or=_or,
    pattern=_pattern,
    default_value=0,
    pattern_mode=DEFAULT_PATTERN_MODE,
):
    if not isinstance(expression, str) or len(expression) == 0:
        return default_value
    regex = resolve_pattern(pattern=pattern, pattern_mode=pattern_mode)
    matches = set(re.findall(regex, expression))
    context = {k: symbol_values.get(k, default_value) for k in matches}
    parsed = _parse_gpr_cached(expression)
    if parsed is None:
        LOGGER.warning(f"Invalid GPR expression (syntax error): {expression}")
        return default_value
    val = _eval_gpr(
        parsed,
        context,
        func_and,
        func_or,
        expression=expression,
        default_value=default_value,
    )
    if val is None:
        val = default_value
    return val


def evaluate_gpr_expression(
    gpr_expressions: list,
    symbol_values: dict,
    func_and=_and,
    func_or=_or,
    default_value=0,
    pattern=_pattern,
    pattern_mode=DEFAULT_PATTERN_MODE,
):
    def evaluate(gpr_expression):
        return evaluate_gpr(
            gpr_expression,
            symbol_values,
            func_and=func_and,
            func_or=func_or,
            default_value=default_value,
            pattern=pattern,
            pattern_mode=pattern_mode,
        )

    results = list(map(evaluate, gpr_expressions))
    return results


def _evaluate_for_symbol_values(args):
    (
        gpr_expressions,
        symbol_values,
        func_and,
        func_or,
        default_value,
        pattern,
        pattern_mode,
    ) = args
    return evaluate_gpr_expression(
        gpr_expressions=gpr_expressions,
        symbol_values=symbol_values,
        func_and=func_and,
        func_or=func_or,
        default_value=default_value,
        pattern=pattern,
        pattern_mode=pattern_mode,
    )


def evaluate_gpr_rules(
    gpr_expressions: list,
    symbol_values_list: list[dict],
    func_and: Callable = _and,
    func_or: Callable = _or,
    default_value=0,
    pattern=_pattern,
    pattern_mode=DEFAULT_PATTERN_MODE,
    n_processes=None,
):
    if n_processes is None or n_processes == 0:
        results = [
            evaluate_gpr_expression(
                gpr_expressions=gpr_expressions,
                symbol_values=sv,
                func_and=func_and,
                func_or=func_or,
                default_value=default_value,
                pattern=pattern,
                pattern_mode=pattern_mode,
            )
            for sv in symbol_values_list
        ]
    else:
        # Use multiprocessing
        if n_processes == -1:
            # Use the minimum of the number of CPUs or the length of symbol_values_list
            n_processes = min(cpu_count(), len(symbol_values_list))

        args = [
            (
                gpr_expressions,
                sv,
                func_and,
                func_or,
                default_value,
                pattern,
                pattern_mode,
            )
            for sv in symbol_values_list
        ]
        with Pool(n_processes) as pool:
            results = pool.map(_evaluate_for_symbol_values, args)

    return results
