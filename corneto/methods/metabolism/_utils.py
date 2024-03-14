import ast
import re
from multiprocessing import Pool, cpu_count

_pattern = r"\b(?!and\b|or\b)[A-Za-z0-9_]+\b"


def _and(a, b):
    if a < 0 and b < 0:
        return max(a, b)
    return min(a, b)


def _or(a, b):
    if a < 0 and b < 0:
        return min(a, b)
    return max(a, b)


def _eval_gpr(node, context, func_and, func_or):
    if isinstance(node, ast.Expression):
        return _eval_gpr(node.body, context, func_and, func_or)
    elif isinstance(node, ast.BoolOp):
        # Process the first value to initialize the result,
        # then iterate through the remaining values if any.
        values_iter = iter(node.values)
        result = _eval_gpr(next(values_iter), context, func_and, func_or)
        for value in values_iter:
            if isinstance(node.op, ast.And):
                result = func_and(result, _eval_gpr(value, context, func_and, func_or))
            elif isinstance(node.op, ast.Or):
                result = func_or(result, _eval_gpr(value, context, func_and, func_or))
        return result
    elif isinstance(node, ast.Name):
        return context[node.id]
    else:
        raise ValueError(f"Unsupported AST node: {type(node).__name__}")


def evaluate_gpr(
    expression: str,
    symbol_values: dict,
    func_and=_and,
    func_or=_or,
    pattern=_pattern,
    default_value=0,
):
    if not isinstance(expression, str):
        return default_value
    matches = set(re.findall(pattern, expression))
    context = {k: symbol_values.get(k, default_value) for k in matches}
    parsed = ast.parse(expression, mode="eval")
    return _eval_gpr(parsed, context, func_and, func_or)


def evaluate_gpr_expression(
    gpr_expressions: list,
    symbol_values: dict,
    func_and=_and,
    func_or=_or,
    default_value=0,
    pattern=_pattern,
):
    def evaluate(gpr_expression):
        return evaluate_gpr(
            gpr_expression,
            symbol_values,
            func_and=func_and,
            func_or=func_or,
            default_value=default_value,
            pattern=pattern,
        )

    results = list(map(evaluate, gpr_expressions))
    return results


def evaluate_gpr_rules(
    gpr_expressions: list,
    symbol_values_list: list[dict],
    func_and=_and,
    func_or=_or,
    default_value=0,
    pattern=_pattern,
    n_processes=None,
):
    def evaluate_for_symbol_values(symbol_values):
        # Assuming evaluate_gpr_expression is properly defined and implemented.
        return evaluate_gpr_expression(
            gpr_expressions=gpr_expressions,
            symbol_values=symbol_values,
            func_and=func_and,
            func_or=func_or,
            default_value=default_value,
            pattern=pattern,
        )

    # Determine the actual number of processes to use
    if n_processes is None or n_processes == 0:
        # Sequential processing without multiprocessing
        results = [evaluate_for_symbol_values(sv) for sv in symbol_values_list]
    else:
        # Use multiprocessing
        if n_processes == -1:
            # Use the minimum of the number of CPUs or the length of symbol_values_list
            n_processes = min(cpu_count(), len(symbol_values_list))

        with Pool(n_processes) as pool:
            results = pool.map(evaluate_for_symbol_values, symbol_values_list)

    return results
